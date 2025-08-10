import re
import random
import numpy as np

from functools import partial

from yeval.task import register_task, YevalTask
from yeval.task.gsm8k import GSM8KTask
from yeval.log.usage import log_logprob

from lang_boot.utils import (
    highest_loglikelihood,
    highest_language_content,
    math_eval_with_postprocessing,
)

def shuffle(dataset, seed=0):
    shuffled_dataset = dataset.shuffle(seed=seed)
    return shuffled_dataset.flatten_indices()

@register_task("gsm8k_train")
class LangBootGSM8KTrainTask(GSM8KTask):
    test_split="train"
    input_text=lambda x: x["question"]
    postprocessor=None
    evaluation={"accuracy": math_eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob

@register_task("json_highest_log_gsm8k_train")
class JSONGSM8KTrainTask(LangBootGSM8KTrainTask):
    data_path="json"
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
    preprocessing=highest_loglikelihood

@register_task("json_highest_lang_gsm8k_train")
class JSONGSM8KTrainTask(LangBootGSM8KTrainTask):
    data_path="json"
    input_text=lambda x: x["input"].split("Translation:")[-1].strip()
    output_text=lambda x: x["output"]
    preprocessing=highest_language_content

@register_task("gsm8k_train_problem")
class ProblemGSM8KTrainTask(LangBootGSM8KTrainTask):
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
