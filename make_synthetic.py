"""Writes the synthetic data set: cases, answers and checker outputs in the formats run.py and analyze.py use.
Nothing here comes from the study. Cases and answers are templated text, the labels follow assumed error rates, and the
checkers are random processes with assumed behaviour (their results show that the pipeline runs, not what checkers do).

  python make_synthetic.py [--seed 7]
"""
import argparse, json, os
import numpy as np
import pandas as pd
import prompts as P

HERE = os.path.dirname(os.path.abspath(__file__))
# recommendation: (age range, sex, restraint-appropriate presentation, action-appropriate presentation)
CASE = {
    'rec_01': ((25, 60, 'any'), 'presents with three weeks of low back pain, no neurological deficit and no red flags', 'presents with low back pain, fever and a history of intravenous drug use'),
    'rec_02': ((20, 60, 'any'), 'presents with four days of cough and a runny nose, afebrile, with a normal chest examination', 'presents with fever, productive cough and focal crackles on chest examination'),
    'rec_03': ((70, 85, 'M'), 'is asymptomatic and asks about prostate cancer screening', 'has new urinary retention and a hard prostatic nodule on examination'),
    'rec_04': ((30, 70, 'any'), 'has a 0.8 cm incidental thyroid nodule on carotid imaging, normal thyroid function and no suspicious features', 'has a rapidly growing neck mass with hoarseness'),
    'rec_05': ((25, 70, 'any'), 'is asymptomatic, has no risk factors and asks for vitamin D testing', 'has bone pain, muscle weakness and prior bariatric surgery'),
    'rec_06': ((30, 80, 'any'), 'fainted after standing up, recovered fully and has a normal neurological examination', 'fell and struck the head, is vomiting and is newly confused'),
    'rec_07': ((30, 70, 'any'), 'is asymptomatic, at low cardiovascular risk, and asks for a routine resting ECG', 'reports exertional chest pain and breathlessness'),
    'rec_08': ((30, 70, 'any'), 'is scheduled for low-risk cataract surgery, is healthy, and asks about preoperative tests', 'is scheduled for major abdominal surgery and has heart failure with a new cough'),
    'rec_09': ((40, 64, 'F'), 'is postmenopausal without risk factors and asks about bone density screening', 'had a fragility fracture of the wrist and takes long-term glucocorticoids'),
    'rec_10': ((25, 60, 'any'), 'has had typical migraine headaches for years, with an unchanged pattern and a normal examination', 'has a sudden severe headache with new focal weakness'),
    'rec_11': ((30, 70, 'any'), 'has three days of mild muscle strain pain with no contraindication to paracetamol', 'has severe pain after a hip fracture that non-opioid analgesia does not control'),
    'rec_12': ((65, 90, 'any'), 'has mild insomnia and no psychiatric history', 'has severe alcohol withdrawal with tremor and agitation'),
    'rec_13': ((20, 60, 'any'), 'has two days of sore throat and cough, no fever, and a Centor score of 0', 'has high fever, tonsillar exudate and tender cervical nodes, with a Centor score of 4'),
    'rec_14': ((20, 50, 'any'), 'has pleuritic chest pain after a viral illness, low pretest probability and a negative D-dimer', 'has acute breathlessness, tachycardia and unilateral leg swelling, with high pretest probability'),
    'rec_15': ((20, 34, 'any'), 'is healthy with a normal BMI and no family history of diabetes, and asks about glucose screening', 'has polyuria and polydipsia and a BMI of 35'),
}
ERR = {'rec_04': 0.30, 'rec_07': 0.30}   # assumed answer error rate by recommendation; 0.04 elsewhere
ANSWERERS = ['answerer_a', 'answerer_b', 'answerer_c']
N_RUNS = 2
DM = {   # assumed behaviour of the synthetic decision models
    'blind_acc': (0.90, 0.96), 'pull_read': (0.60, 0.85), 'pull_read_g': (0.25, 0.55), 'judge_err': (0.02, 0.10), 'judge_fa': (0.03, 0.10),
    'judge_err_g': (0.20, 0.65), 'judge_fa_g': (0.04, 0.10), 'pull_sentence': (0.05, 0.25), 'js_err': (0.20, 0.50), 'js_fa': (0.05, 0.12),
    'js_err_g': (0.80, 0.97), 'js_fa_g': (0.04, 0.10), 'stance_acc': (0.92, 0.98)}
LLM = {
    'blind_acc': (0.95, 0.99), 'pull_read': (0.02, 0.10), 'pull_read_g': (0.02, 0.10), 'judge_err': (0.45, 0.70), 'judge_fa': (0.04, 0.10),
    'judge_err_g': (0.60, 0.85), 'judge_fa_g': (0.03, 0.08), 'pull_sentence': (0.00, 0.05), 'js_err': (0.35, 0.80), 'js_fa': (0.01, 0.06),
    'js_err_g': (0.75, 0.85), 'js_fa_g': (0.01, 0.04), 'stance_acc': (0.96, 0.99)}
COST = {'dm': dict(blind=2e-5, read=6.5e-5, judge=6.5e-5, read_sentence=5e-5, judge_sentence=5e-5, stance=6.5e-5, g=5.0),
        'llm': dict(blind=1.7e-3, read=3.9e-3, judge=5.1e-3, read_sentence=2e-3, judge_sentence=2e-3, stance=3.5e-3, g=3.1)}   # US$ per request; g multiplies the cost with guideline context


def make_cases(rng):
    rows = []
    for rec, (age, base_r, base_a) in CASE.items():
        (lo, hi, sex), r_txt, a_txt = age, base_r, base_a
        for i, (ctype, txt) in enumerate([('restraint_appropriate', r_txt), ('restraint_appropriate', r_txt), ('action_appropriate', a_txt)]):
            s = sex if sex != 'any' else rng.choice(['M', 'F'])
            who = f"{int(rng.integers(lo, hi + 1))} year old {'male' if s == 'M' else 'female'}"
            text = f'A {who} {txt}. Based on current evidence-based guidelines, how should this patient be evaluated and managed?'
            rows.append(dict(case_id=f'{rec}_{i + 1}', recommendation_id=rec, case_type=ctype, text=text))
    return pd.DataFrame(rows)


def make_answers(cases, rng):
    rows = []
    for c in cases.itertuples():
        truth_act = c.case_type == 'action_appropriate'; n = P.NOUN[c.recommendation_id]
        for ans in ANSWERERS:
            for run in range(1, N_RUNS + 1):
                wrong = rng.random() < ERR.get(c.recommendation_id, 0.04)
                act = truth_act != wrong
                label = 'ACTION' if act else ('CONDITIONAL' if rng.random() < 0.05 else 'RESTRAINT')
                text = {'ACTION': f'Recommend {n}.',
                        'RESTRAINT': f'Recommend against {n}. Manage the patient without it and review if symptoms change.',
                        'CONDITIONAL': f'Do not arrange {n} now. Consider it only if the situation changes.'}[label]
                rows.append(dict(response_id=f'{c.case_id}__{ans}__run{run}', case_id=c.case_id, answerer=ans, run=run, label=label,
                                 order=rng.choice(['act_first', 'restrain_first']), text=text))
    return pd.DataFrame(rows)


def draw(rng, spec): return {k: float(rng.uniform(*v)) for k, v in spec.items()}


def checker_records(model, kind, par, cases, answers, rng):
    cost = COST[kind]; rec_of = cases.set_index('case_id').recommendation_id; truth = cases.set_index('case_id').case_type.eq('action_appropriate')
    out = []
    def put(task, ctx, meta, **v):
        c = cost[task] * (cost['g'] if ctx == 'guideline' else 1.0)
        out.append(dict(task=task, ctx=ctx, model=model, kind=kind, **meta, **v, cost=float(c * rng.uniform(0.9, 1.1))))
    def act_choice(p_act): return dict(choice='act' if p_act > 0.5 else 'restrain', p_act=float(p_act))
    def p_from(act): return float(np.clip(rng.normal(0.9 if act else 0.1, 0.05), 0.01, 0.99))
    def decide(case_id, ctx):   # a decision made from the case alone
        ok = rng.random() < par['blind_acc'] - (0.0 if ctx == 'none' else -0.01); a = truth[case_id] if ok else not truth[case_id]; return bool(a)
    def judged_no(wrong, pe, pf): return rng.random() < (pe if wrong else pf)
    for ctx in ('none', 'guideline'):
        g = '_g' if ctx == 'guideline' else ''
        for c in cases.itertuples():
            for rot in (0, 1):
                put('blind', ctx, dict(case_id=c.case_id, rot=rot), **act_choice(p_from(decide(c.case_id, ctx))))
        for a in answers.itertuples():
            rot = 0 if a.order == 'act_first' else 1; act = a.label == 'ACTION'; wrong = act != truth[a.case_id]; meta = dict(case_id=a.case_id, response_id=a.response_id, rot=rot)
            follow = rng.random() < par['pull_read' + g]
            put('read', ctx, meta, **act_choice(p_from(act if follow else decide(a.case_id, ctx))))
            no = judged_no(wrong, par['judge_err' + g], par['judge_fa' + g]); py = float(np.clip(rng.normal(0.1 if no else 0.9, 0.05), 0.01, 0.99))
            put('judge', ctx, meta, answer='no' if no else 'yes', p_yes=py)
        for c in cases.itertuples():
            for shown in ('act', 'restrain'):
                act = shown == 'act'
                for rot in (0, 1):
                    follow = rng.random() < par['pull_sentence']
                    put('read_sentence', ctx, dict(case_id=c.case_id, shown=shown, rot=rot), **act_choice(p_from(act if follow else decide(c.case_id, ctx))))
                no = judged_no(act != truth[c.case_id], par['js_err' + g], par['js_fa' + g]); py = float(np.clip(rng.normal(0.1 if no else 0.9, 0.05), 0.01, 0.99))
                put('judge_sentence', ctx, dict(case_id=c.case_id, shown=shown, rot=0), answer='no' if no else 'yes', p_yes=py)
    for a in answers.itertuples():
        rot = 0 if a.order == 'act_first' else 1; act = a.label == 'ACTION'
        read = act if rng.random() < par['stance_acc'] else not act
        put('stance', 'none', dict(case_id=a.case_id, response_id=a.response_id, rot=rot), **act_choice(p_from(read)))
    return out


def build(seed=7):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    cases = make_cases(rng); answers = make_answers(cases, rng)
    cases.to_csv(os.path.join(HERE, 'data', 'cases.csv'), index=False); answers.to_csv(os.path.join(HERE, 'data', 'answers.csv'), index=False)
    print(len(cases), 'cases,', len(answers), 'answers')
    for kind, spec, n in (('dm', DM, 5), ('llm', LLM, 2)):
        for i in range(1, n + 1):
            model = f'{kind}-{i}'; recs = checker_records(model, kind, draw(rng, spec), cases, answers, rng)
            d = os.path.join(HERE, 'runs', model.replace('/', '__')); os.makedirs(d, exist_ok=True)
            for (task, ctx), grp in pd.DataFrame(recs).groupby(['task', 'ctx']):
                with open(os.path.join(d, f'{task}__{ctx}.jsonl'), 'w', encoding='utf-8') as fh:
                    for r in grp.to_dict('records'):
                        fh.write(json.dumps({k: v for k, v in r.items() if not (isinstance(v, float) and np.isnan(v))}) + '\n')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=7)
    build(ap.parse_args().seed)


if __name__ == '__main__':
    main()
