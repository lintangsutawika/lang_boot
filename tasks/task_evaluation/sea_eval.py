import os
import re
from functools import partial

from yeval.task import register_task, YevalTask

from yeval.log.usage import log_token_usage

path = os.path.dirname(__file__)

def input_text(x, lang="Indonesian"):
    _x = x[lang]
    _input = [_x['question'], "\n".join(_x['choices'])]
    if 'context' in _x:
        _input.insert(0, _x['context'])   
    return "\n".join(_input)

def output_text(x, lang="Indonesian"):
    _x = x[lang]
    answer = _x['answer']
    letter, *text = answer.split(" ")
    text = " ".join(text)
    return [letter, text, answer]

def eval(prediction, ground_truth):
    score = 0
    try:
        letter, text, full_span = ground_truth
        matches = re.findall(r'\((.*?)\)', letter)
        if len(matches) > 0:
            letter = matches[0]
        if full_span == ground_truth:
            return 1
        prediction = prediction.split(".")[0]
        matches = re.findall(r'\((.*?)\)', prediction)
        if len(matches) > 0:
            prediction = matches[0]
        if prediction in ["A", "B", "C", "D", "E"]:
            if prediction == letter:
                score = 1
        elif prediction == text:
            score = 1
    except Exception as e:
        pass
    return score

class CrossMMLUTask(YevalTask):
    data_path="SeaEval/cross_mmlu"
    test_split="test"
    evaluation={"accuracy": eval}

@register_task("sea_eval_cross_mmlu_ind")
class CrossMMLUIndTask(CrossMMLUTask):
    input_text=lambda x: partial(input_text, lang="Indonesian")(x)
    output_text=lambda x: partial(output_text, lang="Indonesian")(x)

@register_task("sea_eval_cross_logiqa_ind")
class CrossLogiqaIndTask(CrossMMLUIndTask):
    data_path="SeaEval/cross_logiqa"

@register_task("sea_eval_cross_mmlu_zho")
class CrossMMLUZhoTask(CrossMMLUTask):
    input_text=lambda x: partial(input_text, lang="Chinese")(x)
    output_text=lambda x: partial(output_text, lang="Chinese")(x)

@register_task("sea_eval_cross_logiqa_zho")
class CrossLogiqaZhoTask(CrossMMLUZhoTask):
    data_path="SeaEval/cross_logiqa"
