"""Send the checker requests and store one JSON line per request.

  python run.py --kind dm  [--models ID,ID] [--tasks blind,read,...] [--contexts none,guideline] [--limit N] [--dry-run]
  python run.py --kind llm [...]

The key is read from the environment variable OPENROUTER_API_KEY. Runs are resumable: requests already stored without an
error are skipped. Output: runs/<model>/<task>__<context>.jsonl, the format analyze.py reads.
"""
import argparse, concurrent.futures as cf, json, os, sys, threading, time
import pandas as pd
import prompts as P

HERE = os.path.dirname(os.path.abspath(__file__))
DM_API = 'https://openrouter.ai/api/alpha/decisions'
LLM_API = 'https://openrouter.ai/api/v1/chat/completions'
DECISION_MODELS = ['liquid/d1-ultra', 'perplexity/pplx-decider-v1.2', 'inception/mercury-decide', 'upstage/solar-decide', 'openai/gpt-6-luna-decisions',
                   'cloudflare/clef', 'togethercomputer/tev1-4b-experimental', 'upstage/solar-decide-flash', 'typesafe/jev-1.14', 'liquid/d1',
                   'typesafe/jev-1.13', 'jaredpalmer/kev-4b', 'cloudflare/clef-omni', 'perplexity/pplx-decider-flash-v1.2',
                   'microsoft/microsoft-decision-1', 'nace-ai/drex-v1.5', 'nace-ai/drex-v2.0', 'cloudflare/clef-flash', 'respan/span-02', 'scaledown/classify-1']
LLMS = ['openai/gpt-6.1-sol', 'anthropic/claude-opus-5.5', 'google/gemini-4-argon', 'deepseek/deepseek-v4.1-flash-1017']
LIST_PRICE = {'anthropic/claude-opus-5.5': (4e-6, 20e-6), 'openai/gpt-6.1-sol': (2e-6, 10e-6), 'google/gemini-4-argon': (2e-6, 10e-6)}
# US$ per input and output token at list price; other leading LLMs (DeepSeek V4.1 Flash) are costed at the price the API reports
TASKS = ['blind', 'read', 'judge', 'read_sentence', 'judge_sentence', 'stance']
_tl = threading.local()


def session():
    import requests
    if not hasattr(_tl, 's'):
        _tl.s = requests.Session()
    return _tl.s


def post(url, body, key, timeout):
    err = None
    for i in range(6):
        try:
            r = session().post(url, headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}, json=body, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            err = f'{r.status_code} {r.text[:200]}'
        except Exception as e:
            err = str(e)
        time.sleep(2 ** i)
    return {'error': err}


def call_dm(model, req, key):
    resp = post(DM_API, P.dm_body(model, req), key, 90)
    if 'error' in resp:
        return {'error': resp['error']}
    try:
        return dict(P.parse_dm(resp, req), raw=resp)
    except Exception as e:
        return {'error': 'parse ' + repr(e), 'raw': resp}


def call_llm(model, req, key):
    body = {'model': model, 'messages': [{'role': 'user', 'content': P.llm_prompt(req)}], 'max_tokens': 6000, 'usage': {'include': True},
            'provider': {'allow_fallbacks': True, 'data_collection': 'deny'}}   # default settings of each model: no reasoning or temperature parameters
    resp = post(LLM_API, body, key, 300)
    if 'error' in resp:
        return {'error': resp['error']}
    text = (resp['choices'][0]['message'].get('content') or '')
    out = P.parse_llm(text, req)
    u = resp.get('usage', {})
    if model in LIST_PRICE:
        pin, pout = LIST_PRICE[model]
        out['cost'] = pin * (u.get('prompt_tokens') or 0) + pout * (u.get('completion_tokens') or 0)
    else:
        out['cost'] = u.get('cost')
    if not out.get('choice') and not out.get('answer'):
        return {'error': 'unparsed reply', 'raw': text}
    return dict(out, raw=text)


def jobs(cases, answers, task, ctx, prefix):
    """All requests of one task and context: (meta, request)."""
    out = []
    rec_of = cases.set_index('case_id').recommendation_id
    text_of = cases.set_index('case_id').text
    if task == 'blind':
        for c in cases.itertuples():
            for rot in (0, 1):
                out.append((dict(case_id=c.case_id, rot=rot), P.build('blind', ctx, c.recommendation_id, c.text, rot=rot, prefix=prefix)))
    elif task in ('read', 'judge', 'stance'):
        for a in answers.itertuples():
            rot = 0 if a.order == 'act_first' else 1   # option order assigned to each answer
            out.append((dict(case_id=a.case_id, response_id=a.response_id, rot=rot),
                        P.build(task, ctx, rec_of[a.case_id], text_of[a.case_id], answer=a.text, rot=rot, prefix=prefix)))
    else:   # one-sentence answer: the sentence for and against the test or treatment, for every case
        for c in cases.itertuples():
            for shown in ('act', 'restrain'):
                for rot in ((0, 1) if task == 'read_sentence' else (0,)):
                    out.append((dict(case_id=c.case_id, shown=shown, rot=rot),
                                P.build(task, ctx, c.recommendation_id, c.text, shown=shown, rot=rot, prefix=prefix)))
    return out


def key_of(m):
    return (m.get('case_id'), m.get('response_id'), m.get('shown'), m.get('rot'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['dm', 'llm'], required=True)
    ap.add_argument('--models'); ap.add_argument('--tasks', default=','.join(TASKS)); ap.add_argument('--contexts', default='none,guideline')
    ap.add_argument('--data', default=os.path.join(HERE, 'data')); ap.add_argument('--runs', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--limit', type=int); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if not os.path.exists(os.path.join(a.data, 'cases.csv')):
        sys.exit(f'No cases in {a.data}. Add cases.csv and answers.csv (formats in the README), or run python analyze.py once to create the example data.')
    cases = pd.read_csv(os.path.join(a.data, 'cases.csv')); answers = pd.read_csv(os.path.join(a.data, 'answers.csv'))
    gpath = os.path.join(a.data, 'guideline_context.txt')
    if 'guideline' in a.contexts.split(',') and not os.path.exists(gpath) and not a.dry_run:
        sys.exit(f'{gpath} not found. Create it from the Choosing Wisely recommendations (see README) or pass --contexts none.')
    prefix = P.guideline_prefix(gpath) if os.path.exists(gpath) else ''
    models = a.models.split(',') if a.models else (DECISION_MODELS if a.kind == 'dm' else LLMS)
    key = None if a.dry_run else os.environ['OPENROUTER_API_KEY']
    call = call_dm if a.kind == 'dm' else call_llm
    for model in models:
        for task in a.tasks.split(','):
            for ctx in a.contexts.split(','):
                if task == 'stance' and ctx == 'guideline':
                    continue   # reading what an answer recommends does not use the guideline context
                js = jobs(cases, answers, task, ctx, prefix)
                path = os.path.join(a.runs, model.replace('/', '__'), f'{task}__{ctx}.jsonl')
                done = set()
                if os.path.exists(path):
                    done = {key_of(r) for r in map(json.loads, open(path)) if 'error' not in r}
                js = [j for j in js if key_of(j[0]) not in done][:a.limit]
                if a.dry_run:
                    print(f'{model} {task} {ctx}: {len(js)} requests to send')
                    if js and a.limit:
                        print(json.dumps(P.dm_body(model, js[0][1]), indent=1) if a.kind == 'dm' else P.llm_prompt(js[0][1]), '\n')
                    continue
                os.makedirs(os.path.dirname(path), exist_ok=True); lock = threading.Lock()
                with cf.ThreadPoolExecutor(a.workers) as ex, open(path, 'a', encoding='utf-8') as fh:
                    futs = {ex.submit(call, model, req, key): meta for meta, req in js}
                    for f in cf.as_completed(futs):
                        rec = dict(task=task, ctx=ctx, model=model, kind=a.kind, **futs[f], **f.result())
                        with lock:
                            fh.write(json.dumps(rec) + '\n'); fh.flush()
                print(model, task, ctx, len(js), 'requests', flush=True)


if __name__ == '__main__':
    main()
