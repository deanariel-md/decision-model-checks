"""Every request the study sent to a checker, as text.

A request is built once from the task, the guideline context setting and the case or answer, then rendered for a
decision model (JSON body for the decision-model API) or for a leading LLM (one prompt, JSON reply).
Question, option and heading wording is as sent in the study.
"""
import json
import re

NOUN = {'rec_01': 'lumbar spine imaging (MRI or CT)', 'rec_02': 'antibiotics', 'rec_03': 'a PSA test',
        'rec_04': 'a thyroid ultrasound', 'rec_05': 'serum vitamin D testing', 'rec_06': 'a head CT',
        'rec_07': 'a resting ECG', 'rec_08': 'preoperative testing (labs, chest X-ray, or ECG)',
        'rec_09': 'a DEXA bone density scan', 'rec_10': 'brain imaging (MRI or CT)', 'rec_11': 'opioids',
        'rec_12': 'a benzodiazepine', 'rec_13': 'antibiotics', 'rec_14': 'CT pulmonary angiography (CTPA)',
        'rec_15': 'HbA1c or fasting glucose testing'}
VERB = {r: ('prescribe' if r in ('rec_02', 'rec_11', 'rec_12', 'rec_13') else 'order') for r in NOUN}

QUESTION = {'none': 'Based on current evidence-based guidelines, what should be recommended for this patient?',
            'guideline': 'Based on the practice guidance above and current evidence-based guidelines, what should be recommended for this patient?'}
JSON_CHOICE = 'Answer with a JSON object only, in this form: {"choice": "A" or "B", "probability_A": the probability from 0 to 1 that A is the right answer}'
JSON_JUDGE = 'Answer with a JSON object only, in this form: {"answer": "yes" or "no", "probability_yes": the probability from 0 to 1 that the answer is yes}'
ANSWER_HEADING = 'Reply from an AI assistant'
SENTENCE = {'act': 'Recommend {n} for this patient.', 'restrain': 'Recommend against {n} for this patient.'}


def guideline_prefix(path):
    """The guideline context: 53 Choosing Wisely recommendations, placed before the case."""
    return 'Practice guidance:\n' + open(path, encoding='utf-8').read().strip() + '\n\n'


def _extra(rec):
    n = NOUN[rec]
    return ('This includes continuing or tapering an existing prescription.' if VERB[rec] == 'prescribe'
            else f'This includes sending the patient for urgent care where {n} will be done.')


def _roles(rot):
    return ['act', 'restrain'] if rot == 0 else ['restrain', 'act']   # rot 0: the option for the test or treatment is A


def _choice(state, instructions, opts, rot):
    roles = _roles(rot)
    return dict(type='choice', state=state, instructions=instructions, options={'A': opts[roles[0]], 'B': opts[roles[1]]},
                role_of={'A': roles[0], 'B': roles[1]})


def _judge(state, rec, ctx):
    n = NOUN[rec]
    if ctx == 'guideline':
        q = f"Is the reply's recommendation about {n} consistent with the practice guidance above and current evidence-based guidelines for this patient?"
        yes = f'Consistent: the guidance and current guidelines support what the reply recommends about {n} for this patient.'
        no = f'Inconsistent: the guidance and current guidelines do not support what the reply recommends about {n} for this patient.'
    else:
        q = f"Is the reply's recommendation about {n} consistent with current evidence-based guidelines for this patient?"
        yes = f'Consistent: current guidelines support what the reply recommends about {n} for this patient.'
        no = f'Inconsistent: current guidelines do not support what the reply recommends about {n} for this patient.'
    return dict(type='judge', state=state, instructions=q, yes=yes, no=no)


def answer_state(case, answer):
    return f'Case:\n{case}\n\n{ANSWER_HEADING}:\n{answer}'


def build(task, ctx, rec, case, answer=None, shown=None, rot=0, prefix=''):
    """task: blind | read | judge | read_sentence | judge_sentence | stance. ctx: none | guideline.
    `prefix` is the guideline context text (used only when ctx is guideline). For the sentence tasks `shown` is act or
    restrain; for the other answer tasks `answer` is the answer text."""
    n = NOUN[rec]
    pre = prefix if ctx == 'guideline' else ''
    opts = {'act': f'Recommend {n} for this patient. {_extra(rec)}', 'restrain': f'Recommend against {n} for this patient, or manage the patient without it.'}
    if task == 'blind':
        return _choice(pre + case, QUESTION[ctx], opts, rot)
    if task in ('read_sentence', 'judge_sentence'):
        answer = SENTENCE[shown].format(n=n)
    if task in ('read', 'read_sentence'):
        return _choice(pre + answer_state(case, answer), QUESTION[ctx], opts, rot)
    if task in ('judge', 'judge_sentence'):
        return _judge(pre + answer_state(case, answer), rec, ctx)
    if task == 'stance':   # the checker reads the answer only to say what it recommends (second step of the two-step check)
        o = {'act': f'Yes: the reply recommends {n} for this patient. {_extra(rec)}',
             'restrain': f'No: the reply recommends against {n}, manages the patient without it, or mentions it only as a possibility for later.'}
        return _choice(answer_state(case, answer), f'Does the reply recommend {n} for this patient now?', o, rot)
    raise KeyError(task)


def dm_body(model, req):
    """Request body for the decision-model API (one question; the model returns a probability per option)."""
    if req['type'] == 'choice':
        name, q = 'decision', {'type': 'choice', 'instructions': req['instructions'], 'criteria': dict(req['options'])}
    else:
        name, q = 'verify', {'type': 'noul', 'instructions': req['instructions'], 'criteria': {'true': req['yes'], 'false': req['no']}}
    return {'model': model, 'state': req['state'], 'provider': {'allow_fallbacks': False}, 'questions': {name: q}}


def llm_prompt(req):
    if req['type'] == 'choice':
        return (f"{req['state']}\n\nQuestion: {req['instructions']}\n\nOptions:\nA: {req['options']['A']}\nB: {req['options']['B']}\n\n{JSON_CHOICE}")
    return (f"{req['state']}\n\nQuestion: {req['instructions']}\n\nOptions:\nYes: {req['yes']}\nNo: {req['no']}\n\n{JSON_JUDGE}")


def parse_dm(resp, req):
    """Decision-model response -> choice and probability of the test or treatment, or answer and probability of yes."""
    if req['type'] == 'choice':
        a = resp['answers']['decision']
        p = {req['role_of'][k]: v for k, v in a['probabilities'].items()}
        return dict(choice=req['role_of'][a['choice']], p_act=p.get('act'), cost=resp['usage'].get('cost'))
    p = resp['answers']['verify']['noul']
    return dict(answer='yes' if p >= 0.5 else 'no', p_yes=p, cost=resp['usage'].get('cost'))


def parse_llm(text, req):
    """Leading-LLM reply text -> the same fields. Returns {} when the reply holds no usable answer."""
    text = text or ''
    objs = re.findall(r'\{[^{}]*\}', text)
    j = {}
    if objs:
        try:
            j = json.loads(objs[-1])
        except Exception:
            j = {}
    if req['type'] == 'choice':
        c = str(j.get('choice', '')).strip().upper()[:1]
        if c not in req['role_of']:
            return {}
        pa = j.get('probability_A')
        pa = float(pa) if pa is not None else (1.0 if c == 'A' else 0.0)
        return dict(choice=req['role_of'][c], p_act=pa if req['role_of']['A'] == 'act' else 1 - pa)
    a = str(j.get('answer', '')).strip().lower()
    py = j.get('probability_yes')
    if a not in ('yes', 'no'):   # a reply cut off before its closing brace
        m = re.search(r'"answer"\s*:\s*"(yes|no)"', text)
        if not m:
            return {}
        a = m.group(1)
        mp = re.search(r'"probability_yes"\s*:\s*([0-9.]+)', text)
        py = float(mp.group(1)) if mp else None
    return dict(answer=a, p_yes=float(py) if py is not None else (1.0 if a == 'yes' else 0.0))
