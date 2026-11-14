"""Scores the stored checker outputs and writes Tables 1 and 2, the shift towards the answer and the paired differences.

  python analyze.py [--data data] [--runs runs] [--out results] [--nboot 2000]

Flags (per answer and checker)
  blind, read, read_sentence   flagged when the checker's decision (test or treatment: yes or no) differs from the answer's recommendation label
  judge, judge_sentence        flagged when the checker did not judge the answer consistent with guidelines (a missing reply counts as a flag)
  twostep                      flagged when the checker's blind decision differs from the recommendation it read from the answer
A blind decision is made once per case and option order and compared with every answer to that case in the answer's option order.
Errors flagged: share of wrong answers flagged. False alarms: share of correct answers flagged.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CTXS = ['none', 'guideline']
ROWS = [('blind', 'Blind decision'), ('read', 'Decision after reading'), ('judge', 'Judging the answer'),
        ('read_sentence', 'Decision after reading a one-sentence answer'), ('judge_sentence', 'Judging a one-sentence answer'), ('twostep', 'Two-step check')]
KINDS = [('dm', 'decision models'), ('llm', 'leading LLMs')]
PAIRS = [('read', 'blind'), ('judge', 'blind'), ('judge', 'read'), ('read_sentence', 'read'), ('judge_sentence', 'judge'), ('twostep', 'blind')]
SHIFT_TASKS = ['read', 'judge', 'read_sentence', 'judge_sentence']


def load(data, runs):
    cases = pd.read_csv(os.path.join(data, 'cases.csv')); answers = pd.read_csv(os.path.join(data, 'answers.csv'))
    A = answers.merge(cases[['case_id', 'recommendation_id', 'case_type']], on='case_id')
    A['act'] = (A.label == 'ACTION').astype(int)                      # no clear recommendation counts as not recommending
    A['truth'] = (A.case_type == 'action_appropriate').astype(int)
    A['err'] = (A.act != A.truth).astype(int)                         # overuse or false restraint
    A['rot'] = (A.order != 'act_first').astype(int)
    A['shown'] = np.where(A.act == 1, 'act', 'restrain')
    rows = []
    for f in sorted(glob.glob(os.path.join(runs, '*', '*.jsonl'))):
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            if 'error' not in r:
                r.pop('raw', None); rows.append(r)
    R = pd.DataFrame(rows)
    for c in ['case_id', 'response_id', 'shown', 'rot', 'choice', 'answer', 'p_act', 'p_yes', 'cost']:
        if c not in R:
            R[c] = np.nan
    # decision of a checker: a leading LLM's stated choice; a decision model's more probable option (a tie is not flagged);
    # the choice stored for a decision model is used where its probability is missing and for the stance question
    R['dec'] = R.choice
    dm = (R.kind == 'dm') & R.p_act.notna() & (R.task != 'stance')
    R.loc[dm, 'dec'] = np.select([R.p_act[dm] > 0.5, R.p_act[dm] < 0.5], ['act', 'restrain'], 'tie')
    return A.reset_index(drop=True), R


def decision_flag(x, col):
    f = ((x[col] == 'act').astype(int) != x.act).astype(float)
    f[x[col] == 'tie'] = 0.0
    return f.where(x[col].notna())


def model_flags(A, R):
    """Flags of one checker: {(task, ctx): Series of 0/1/NaN indexed by response_id}."""
    out = {}
    for ctx in CTXS:
        def sub(t):
            s = R[(R.task == t) & (R.ctx == ctx)]
            return s if len(s) else None
        b = sub('blind')
        if b is not None:
            d = b.drop_duplicates(['case_id', 'rot'], keep='last')[['case_id', 'rot', 'dec']].rename(columns={'dec': 'bc'})
            x = A.merge(d, on=['case_id', 'rot'], how='left')
            out[('blind', ctx)] = pd.Series(decision_flag(x, 'bc').values, index=A.response_id)
            out[('blind_acc', ctx)] = pd.Series(((x.bc == 'act').astype(int) == x.truth).astype(float).where(x.bc.notna()).values, index=A.response_id)
        for t in ('read', 'judge'):
            s = sub(t)
            if s is not None:
                d = s.drop_duplicates('response_id', keep='last').set_index('response_id')
                if t == 'read':
                    x = A.assign(c=A.response_id.map(d.dec)); out[(t, ctx)] = pd.Series(decision_flag(x, 'c').values, index=A.response_id)
                    out[('read_acc', ctx)] = pd.Series(((x.c == 'act').astype(int) == x.truth).astype(float).where(x.c.notna()).values, index=A.response_id)
                else:
                    a = A.response_id.map(d.answer)
                    out[(t, ctx)] = pd.Series((a != 'yes').astype(float).values, index=A.response_id)
        s = sub('read_sentence')
        if s is not None:
            d = s.drop_duplicates(['case_id', 'shown', 'rot'], keep='last')[['case_id', 'shown', 'rot', 'dec']].rename(columns={'dec': 'c'})
            x = A.merge(d, on=['case_id', 'shown', 'rot'], how='left'); out[('read_sentence', ctx)] = pd.Series(decision_flag(x, 'c').values, index=A.response_id)
        s = sub('judge_sentence')
        if s is not None:
            d = s.drop_duplicates(['case_id', 'shown'], keep='last')[['case_id', 'shown', 'answer']].rename(columns={'answer': 'a'})
            x = A.merge(d, on=['case_id', 'shown'], how='left'); out[('judge_sentence', ctx)] = pd.Series((x.a != 'yes').astype(float).values, index=A.response_id)
    st = R[(R.task == 'stance') & (R.ctx == 'none')]
    if len(st):
        d = st.drop_duplicates('response_id', keep='last').set_index('response_id')
        read_act = A.response_id.map(d.choice).values
        for ctx in CTXS:
            if ('blind', ctx) in out:
                fb = out[('blind', ctx)].values
                b_act = np.where(fb == 1, 1 - A.act.values, A.act.values)
                fl = ((b_act != (read_act == 'act').astype(int)) | pd.isna(read_act)).astype(float)
                out[('twostep', ctx)] = pd.Series(np.where(np.isnan(fb), np.nan, fl), index=A.response_id)
    return out


def cost_of(R, task, ctx):
    s = R[(R.task == task) & (R.ctx == ctx)].cost
    return float(s.mean()) if s.notna().any() else np.nan


def pct(x): return f'{100 * x:.1f}'


def cell(v, kind):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if not len(v): return 'NA'
    if kind == 'dm':
        q = np.percentile(v, [50, 25, 75]); return f'{pct(q[0])} ({pct(q[1])}-{pct(q[2])})'
    return pct(v[0]) if v.min() == v.max() else f'{pct(v.min())}-{pct(v.max())}'


def table1(M):
    head = ['Task'] + [f'{m} ({c})' for c in CTXS for m in ('Errors flagged, %', 'False alarms, %')]
    L = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for kind, name in KINDS:
        d = M[M.kind == kind]
        if d.empty: continue
        n = d.model.nunique()
        L.append(f'| **{n} {name}, ' + ('median (IQR)' if kind == 'dm' else 'range') + '** |' + ' |' * 4)
        for t, label in ROWS:
            cells = []
            for ctx in CTXS:
                x = d[(d.task == t) & (d.ctx == ctx)]
                cells += [cell(x.sens, kind), cell(x.fa, kind)] if len(x) else ['NA', 'NA']
            L.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(L)


def table2(M, A, R):
    prev = A.err.mean(); ne = 1000 * prev
    head = ['Check'] + [f'{m} ({c})' for c in CTXS for m in ('Answers flagged', 'Errors flagged', 'Model cost, US$')]
    L = [f'Per 1000 answers holding {ne:.0f} errors (the error rate of these answers, {100 * prev:.1f}%).', '',
         '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for t, label in [r for r in ROWS if r[0] in ('blind', 'read', 'judge', 'twostep')]:
        L.append(f'| **{label}** |' + ' |' * 6)
        for kind, name in KINDS:
            d = M[(M.kind == kind) & (M.task == t)]
            if d.empty: continue
            cells = []
            for ctx in CTXS:
                x = d[d.ctx == ctx]
                if x.empty: cells += ['NA'] * 3; continue
                fl = 1000 * (prev * x.sens + (1 - prev) * x.fa)
                costs = []
                for m in x.model:
                    Rm = R[R.model == m]
                    c = (cost_of(Rm, 'blind', ctx) + cost_of(Rm, 'stance', 'none')) if t == 'twostep' else cost_of(Rm, t, ctx)
                    costs.append(1000 * c)
                c = np.nanmean(costs) if not np.all(np.isnan(costs)) else np.nan
                cells += [f'{fl.mean():.0f}', f'{ne * x.sens.mean():.0f}', 'NA' if np.isnan(c) else f'{c:.3f}' if c < 0.1 else f'{c:.2f}']
            L.append(f'| Average {name[:-1]} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(L)


def pooled_diff(Fa, Fb, A, nboot, rng):
    """Mean over checkers of (errors flagged in a) minus (errors flagged in b); 95% CI from bootstrap resamples of cases."""
    e = A[A.err == 1]; ms = [m for m in Fa.columns if m in Fb.columns]
    a = Fa.reindex(e.response_id)[ms].values; b = Fb.reindex(e.response_id)[ms].values
    ok = ~(np.isnan(a) | np.isnan(b)); d = np.where(ok, a - b, 0.0)
    cases, inv = np.unique(e.case_id.values, return_inverse=True)
    dc = np.bincount(inv, weights=d.sum(1), minlength=len(cases)); nc = np.bincount(inv, weights=ok.sum(1), minlength=len(cases))
    est = dc.sum() / nc.sum(); bs = []
    for _ in range(nboot):
        w = np.bincount(rng.integers(0, len(cases), len(cases)), minlength=len(cases)); bs.append((w * dc).sum() / (w * nc).sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    fewer = int(((np.nanmean(np.where(ok, a, np.nan), 0)) < (np.nanmean(np.where(ok, b, np.nan), 0))).sum())
    return dict(n_checkers=len(ms), a=np.nanmean(np.where(ok, a, np.nan)), b=np.nanmean(np.where(ok, b, np.nan)), diff=est, lo=lo, hi=hi, n_first_lower=fewer)


def shifts(A, F, nboot, rng):
    """In cases with answers on both sides, the share of checker decisions recommending the test or treatment after an answer
    recommending it, minus after an answer advising against it (judging an answer correct counts as agreeing with it)."""
    a = A[A.label.isin(['ACTION', 'RESTRAINT'])]
    mm = a.groupby('case_id').act.agg(['min', 'max']); mixed = mm[(mm['min'] == 0) & (mm['max'] == 1)].index
    a = a[a.case_id.isin(mixed)]; rows = []
    for (kind, t, ctx), fl in F.items():
        if t not in SHIFT_TASKS: continue
        f = fl.reindex(a.response_id); cact = pd.DataFrame(np.where(f.values == 1, 1 - a.act.values[:, None], a.act.values[:, None]), columns=f.columns)
        cact = cact.where(f.notna().values); cact['case_id'] = a.case_id.values; cact['act'] = a.act.values
        g = cact.groupby(['case_id', 'act']).mean(); D = (g.xs(1, level='act') - g.xs(0, level='act')).values   # cases x checkers
        est = np.nanmean(np.nanmean(D, 0)); bs = [np.nanmean(np.nanmean(D[rng.integers(0, len(D), len(D))], 0)) for _ in range(nboot)]
        lo, hi = np.percentile(bs, [2.5, 97.5]); per = np.nanmean(D, 0)
        rows.append(dict(kind=kind, task=t, ctx=ctx, n_cases=len(D), shift=100 * est, lo=100 * lo, hi=100 * hi, min=100 * per.min(), max=100 * per.max()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(HERE, 'data')); ap.add_argument('--runs', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results')); ap.add_argument('--nboot', type=int, default=2000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(20261021)
    if not glob.glob(os.path.join(a.runs, '*', '*.jsonl')) or not os.path.exists(os.path.join(a.data, 'cases.csv')):
        if (a.data, a.runs) != (os.path.join(HERE, 'data'), os.path.join(HERE, 'runs')):
            raise SystemExit(f'No checker outputs found in {a.runs} or no cases in {a.data}.')
        import make_synthetic   # nothing to score yet: create the example data on the spot
        make_synthetic.build()
    A, R = load(a.data, a.runs)
    print(f'{len(A)} answers, {int(A.err.sum())} wrong ({100 * A.err.mean():.1f}%), {A.case_id.nunique()} cases, {R.model.nunique()} checkers')
    rows, F = [], {}
    for model, Rm in R.groupby('model'):
        kind = Rm.kind.iloc[0]; fl = model_flags(A, Rm); e = (A.set_index('response_id').err == 1)
        for (t, ctx), s in fl.items():
            if t.endswith('_acc'): continue
            F.setdefault((kind, t, ctx), {})[model] = s
            r = dict(model=model, kind=kind, task=t, ctx=ctx, n=int(s.notna().sum()), coverage=float(s.notna().mean()),
                     sens=float(s[e].mean()), fa=float(s[~e].mean()))
            if (t + '_acc', ctx) in fl: r['accuracy'] = float(fl[(t + '_acc', ctx)].mean())
            rows.append(r)
    M = pd.DataFrame(rows); M.to_csv(os.path.join(a.out, 'checker_metrics.csv'), index=False)
    if (M.coverage < 1).any(): print('Warning: incomplete coverage for', M[M.coverage < 1][['model', 'task', 'ctx', 'coverage']].to_string(index=False))
    F = {k: pd.DataFrame(v) for k, v in F.items()}
    t1, t2 = table1(M), table2(M, A, R)
    open(os.path.join(a.out, 'table1.md'), 'w', encoding='utf-8').write(
        '**Table 1: Errors flagged and false alarms, by checker, task and guideline context**\n\n' + t1 + '\n')
    open(os.path.join(a.out, 'table2.md'), 'w', encoding='utf-8').write(
        '**Table 2: Answers flagged, errors flagged and model cost per 1000 answers**\n\n' + t2 + '\n')
    D = []
    for kind, _ in KINDS:
        for ctx in CTXS:
            for t1_, t2_ in PAIRS:
                if (kind, t1_, ctx) in F and (kind, t2_, ctx) in F:
                    D.append(dict(kind=kind, ctx=ctx, comparison=f'{t1_} - {t2_}', **pooled_diff(F[(kind, t1_, ctx)], F[(kind, t2_, ctx)], A, a.nboot, rng)))
        for t in ('blind', 'read', 'judge'):
            if (kind, t, 'guideline') in F and (kind, t, 'none') in F:
                D.append(dict(kind=kind, ctx='guideline - none', comparison=t, **pooled_diff(F[(kind, t, 'guideline')], F[(kind, t, 'none')], A, a.nboot, rng)))
    D = pd.DataFrame(D)
    for c in ['a', 'b', 'diff', 'lo', 'hi']: D[c] = 100 * D[c]
    D.rename(columns={'a': 'errors_flagged_first', 'b': 'errors_flagged_second', 'diff': 'difference_pp', 'lo': 'ci_low', 'hi': 'ci_high'}).to_csv(os.path.join(a.out, 'differences.csv'), index=False, float_format='%.1f')
    S = shifts(A, F, a.nboot, rng); S.to_csv(os.path.join(a.out, 'shifts.csv'), index=False, float_format='%.1f')
    print('\n' + t1 + '\n\n' + t2 + '\n\nPaired differences in errors flagged (percentage points)\n' + D.to_string(index=False, float_format=lambda v: f'{v:.1f}'))
    print('\nShift towards the answer (percentage points)\n' + S.to_string(index=False, float_format=lambda v: f'{v:.1f}'))


if __name__ == '__main__':
    main()
