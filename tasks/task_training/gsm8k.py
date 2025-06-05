import random
import numpy as np

from functools import partial
from yeval.task import register_task, YevalTask
from yeval.metrics import math_eval

from yeval.task.gsm8k import GSM8KTask
from yeval.response.math_responses import get_boxed_answer

def shuffle(dataset, seed=0):
    shuffled_dataset = dataset.shuffle(seed=seed)
    return shuffled_dataset.flatten_indices()

def clean(x):
    if x.startswith("\\$"):
        x.replace("\\$", "")
    return x

@register_task("gsm8k")
class LangBootGSM8KTask(GSM8KTask):
    input_text=lambda x: x["question"]
    test_split="train"
    postprocessing=clean
    postprocessor=get_boxed_answer
    sample_agg_fn={
        "acc_agg": np.mean,
        "acc": lambda x: x,
        }
    evaluation={
        "acc_agg": math_eval,
        "acc": math_eval,
        }

if __name__ == "__main__":
    pass
