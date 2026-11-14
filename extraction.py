"""Taking the test or treatment from the answer (a sensitivity analysis).

A deployed checker would not be told which test or treatment an answer is about. In this analysis an LLM listed every test,
imaging study, procedure or medication the answer discusses, and a listed item matched the case's test or treatment when it
contained one of the terms below (lower case, substring match). Each checker then made a blind decision on the first matching
item, worded as listed, with the options of the main blind decision. Answers without a match were counted as not flagged.
"""
import json
import re

import prompts as P

REQUEST = ('{state}\n\nQuestion: List every test, imaging study, procedure, or medication that this reply discusses for this patient, '
           'whether or not the reply recommends it. Name each in a few words, and do not say whether it is recommended.\n\n'
           'Answer with a JSON object only, in this form: {{"interventions": ["first", "second", "..."]}}')

TERMS = {
    'rec_01': ['mri', 'ct', 'x-ray', 'xray', 'x ray', 'radiograph', 'imaging'],
    'rec_02': ['antibiotic', 'amoxicillin', 'azithromycin', 'penicillin', 'doxycycline', 'cef', 'augmentin'],
    'rec_03': ['psa', 'prostate-specific', 'prostate specific', 'prostate cancer screening'],
    'rec_04': ['ultrasound', 'sonograph'],
    'rec_05': ['vitamin d', '25-oh', '25(oh)', '25-hydroxy'],
    'rec_06': ['ct', 'computed tomography', 'neuroimaging', 'head imaging', 'brain imaging'],
    'rec_07': ['ecg', 'ekg', 'electrocardiogra'],
    'rec_08': ['preoperative', 'pre-operative', 'laborator', 'labs', 'chest x-ray', 'ecg', 'electrocardiogra', 'testing'],
    'rec_09': ['dexa', 'dxa', 'bone density', 'densitometry'],
    'rec_10': ['mri', 'ct', 'neuroimaging', 'imaging'],
    'rec_11': ['opioid', 'oxycodone', 'hydrocodone', 'tramadol', 'morphine', 'codeine', 'tapentadol'],
    'rec_12': ['benzodiazepine', 'lorazepam', 'alprazolam', 'diazepam', 'clonazepam', 'temazepam', 'chlordiazepoxide'],
    'rec_13': ['antibiotic', 'amoxicillin', 'azithromycin', 'penicillin', 'doxycycline', 'cef', 'augmentin'],
    'rec_14': ['ctpa', 'ct pulmonary', 'ct angiogra', 'pulmonary angiogra', 'ct chest', 'computed tomography pulmonary'],
    'rec_15': ['a1c', 'glucose', 'diabetes screening', 'diabetes testing', 'screening for diabetes'],
}


def request(case, answer):
    """The listing request for one answer, sent as one user message to the LLM."""
    return REQUEST.format(state=P.answer_state(case, answer))


def items(reply_text):
    """The listed items from the reply, or [] when the reply holds no readable list."""
    objs = re.findall(r'\{[^{}]*\}', reply_text or '')
    try:
        return list(json.loads(objs[-1]).get('interventions') or []) if objs else []
    except Exception:
        return []


def first_match(listed, rec):
    """The first listed item that names the case's test or treatment, as listed, or None."""
    hit = [str(i).strip().rstrip('.') for i in listed if any(k in str(i).lower() for k in TERMS[rec])]
    return hit[0] if hit else None


def blind_request(rec, case, phrase, rot=0):
    """The blind decision on the matched item: the main blind decision with the item named as listed."""
    extra = ('This includes continuing or tapering an existing prescription.' if P.VERB[rec] == 'prescribe'
             else f'This includes sending the patient for urgent care where {phrase} will be done.')
    opts = {'act': f'Recommend {phrase} for this patient. {extra}',
            'restrain': f'Recommend against {phrase} for this patient, or manage the patient without it.'}
    return P._choice(case, P.QUESTION['none'], opts, rot)
