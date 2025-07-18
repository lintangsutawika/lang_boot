import os
import re
from functools import partial

from yeval.task import register_task, YevalTask

from yeval.log.usage import log_token_usage

path = os.path.dirname(__file__)

def input_text(x):
    _x = x['Indonesian']
    _input = [_x['question'], "\n".join(_x['choices'])]
    if 'context' in _x:
        _input.insert(0, _x['context'])   
    return "\n".join(_input)

def output_text(x):
    _x = x['Indonesian']
    answer = _x['answer']
    letter, *text = answer.split(" ")
    text = " ".join(text)
    return [letter, text, answer]

def eval_fn(prediction, ground_truth):
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

@register_task("belebele_ind")
class MGSMTask(YevalTask):
    data_path="facebook/belebele"
    data_name="ind_Latn"
    input_text=input_text
    output_text=output_text
    test_split="test"
    evaluation={"accuracy": eval_fn}
