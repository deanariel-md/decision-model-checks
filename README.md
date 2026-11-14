# Blind AI second opinions on clinical LLM answers

Code to test AI checkers on LLM answers to clinical cases. Twenty decision models and four leading LLMs each check every answer in three ways: deciding the case blind, deciding it after reading the answer, and judging the answer. Every check runs with and without guideline context. The outputs are errors flagged, false alarms, the shift towards the answer, and cost.

The data in this repository are synthetic. They show that the pipeline runs end to end and fix the file formats, and they carry no result of the study. The study's cases and the recommendation label of each answer are public at https://doi.org/10.17605/OSF.IO/8QUG5.

## Quick start

```
pip install -r requirements.txt
python analyze.py                  # creates the example data on the spot, scores it, writes results/
python run.py --kind dm --dry-run  # then counts the requests to send, sends none
```

`analyze.py` calls `make_synthetic.py` when `data/` and `runs/` do not exist yet (seeded, a few seconds); `python make_synthetic.py` runs it alone.

To run the experiment on other cases and answers, replace `data/cases.csv` and `data/answers.csv`, set `OPENROUTER_API_KEY`, and run `python run.py --kind dm` and `python run.py --kind llm`. Options are `--models`, `--tasks`, `--contexts` and `--limit`. Interrupted runs resume. Study data in another layout need converting to the formats below before `analyze.py` can score them.

## Layout

| Path | Contents |
|---|---|
| `prompts.py` | The questions, options and prompts of the main tasks, as sent, with the request renderers and reply parsers |
| `run.py` | Sends the requests and stores one JSON line per request |
| `analyze.py` | Scores the outputs and writes the results tables, the shifts and the paired differences |
| `extraction.py` | The request and term lists used to take the test or treatment from an answer (a sensitivity analysis) |
| `make_synthetic.py` | Writes the example `data/` (`cases.csv`, `answers.csv`) and `runs/<checker>/<task>__<context>.jsonl` |
| `results/` | Written by `analyze.py` |

## Checks and flags

| Task | Checker sees | Asked | Flagged when |
|---|---|---|---|
| `blind` | the case | what to recommend | its decision differs from the answer's label |
| `read` | the case and the answer | what to recommend | its decision differs from the answer's label |
| `judge` | the case and the answer | whether the answer is consistent with guidelines | it judges the answer inconsistent, or does not reply |
| `read_sentence`, `judge_sentence` | the case and a one-sentence answer | as `read`, `judge` | as `read`, `judge` |
| `stance` | the case and the answer | whether the answer recommends the test or treatment | in the two-step check, the blind decision differs from this reading |

A blind decision is made once per case and option order and compared with every answer to that case in the answer's option order. A decision model's decision is its more probable option, and a tie is not flagged. An answer is wrong when it recommends the test or treatment in a restraint case (overuse) or does not in an action case (false restraint).

Errors flagged is the share of wrong answers flagged, and false alarms is the share of correct answers flagged. Differences are means across the checkers of each kind, with 95% CIs from case-level bootstrap resamples (seed fixed in `analyze.py`). Leading-LLM cost is priced from token counts at list prices, except for DeepSeek V4.1 Flash, whose cost is the one the API reports, as for the decision models. The checkers in `run.py` are those of the study. The study sent some leading-LLM requests through the developers' own APIs with the same request text.

## Formats

`cases.csv`: `case_id`, `recommendation_id` (`rec_01` to `rec_15`), `case_type` (`restraint_appropriate` or `action_appropriate`), `text`.

`answers.csv`: `response_id`, `case_id`, `answerer`, `run`, `label` (`ACTION`, `RESTRAINT` or `CONDITIONAL`), `order` (`act_first` or `restrain_first`, the option order a checker sees), `text`. The label is the answer's recommendation, and `CONDITIONAL` counts as not recommending.

Checker outputs, one JSON object per line: `task`, `ctx` (`none` or `guideline`), `model`, `kind` (`dm` or `llm`), the request key (`case_id`, `response_id`, `shown`, `rot`), and the reply: `choice` and `p_act` for decision tasks, `answer` and `p_yes` for judging tasks, plus `cost`.

## Guideline context

The guideline condition places 53 recommendations before each question: the 15 the cases are built on, in the benchmark's wording, and 38 from Choosing Wisely Canada's family medicine, internal medicine and geriatrics lists (https://choosingwiselycanada.org/recommendations/). The paper's supplementary appendix lists all 53 in the order sent. The text belongs to its publishers and is not included. To run that condition, save the recommendations as `data/guideline_context.txt`, one per line, numbered `1.` to `53.`, in that order. The synthetic example and `analyze.py` do not need it.

## Other analyses

The robustness variants (other answer headings, a warning that the answer may be wrong, other option wordings) are worded in the paper's supplementary appendix and are not part of this code.
