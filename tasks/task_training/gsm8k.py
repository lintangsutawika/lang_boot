import random
import numpy as np

from functools import partial
from yeval.task import register_task, YevalTask
from yeval.metrics import math_eval

from yeval.task.gsm8k import GSM8KTask
from yeval.response.math_responses import get_boxed_answer

from yeval.metrics import math_eval

def shuffle(dataset, seed=0):
    shuffled_dataset = dataset.shuffle(seed=seed)
    return shuffled_dataset.flatten_indices()

def clean(x):
    if x.startswith("\\$"):
        x.replace("\\$", "")
    return x

def math_eval_with_postprocessing(x, y):
    """
    Evaluates the math problem and returns the accuracy.
    """
    x = get_boxed_answer(x)
    x = clean(x)
    
    return math_eval(x, y)

@register_task("gsm8k_train")
class LangBootGSM8KTrainTask(GSM8KTask):
    test_split="train"
    postprocessor=None
    evaluation={"accuracy": math_eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}

@register_task("gsm8k_train_problem")
class OpenR1Math220KTask(LangBootGSM8KTrainTask):
    input_text=lambda x: x["question"]
    evaluation=None

@register_task("gsm8k_train_solution")
class OpenR1Math220KTask(LangBootGSM8KTrainTask):
    input_text=lambda x: x["answer"]
    evaluation=None

@register_task("gsm8k_test")
class LangBootGSM8KTestTask(GSM8KTask):
    test_split="test"
    # postprocessor=get_boxed_answer
    postprocessor=None
    evaluation={"accuracy": math_eval_with_postprocessing}

@register_task("gsm8k_test_problem")
class OpenR1Math220KTask(LangBootGSM8KTrainTask):
    input_text=lambda x: x["question"]
    evaluation=None

@register_task("gsm8k_test_solution")
class OpenR1Math220KTask(LangBootGSM8KTrainTask):
    input_text=lambda x: x["answer"]
    evaluation=None

if __name__ == "__main__":
    pass
