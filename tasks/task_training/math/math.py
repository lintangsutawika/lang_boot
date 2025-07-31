import os
import re
import random
import numpy as np

from functools import partial

from yeval.task import register_task, YevalTask
from yeval.log.usage import log_logprob
from yeval.response.math_responses import get_boxed_answer

from lang_boot.utils import (
    highest_loglikelihood,
    highest_language_content,
    math_eval_with_postprocessing,
)

path = os.path.dirname(__file__)

@register_task("math_train")
class BaseMATHTrainTask(YevalTask):
    data_path="json"
    data_name=None
    data_kwargs={"data_files": {"train": os.path.join(path, "math.jsonl")}}
    input_text=lambda x: x["problem"]
    output_text=lambda x: get_boxed_answer(x["solution"])
    test_split="train"
    evaluation={"accuracy": math_eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob

@register_task("json_highest_log_math_train")
class JSONMATHTrainTask(BaseMATHTrainTask):
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
    preprocessing=highest_loglikelihood

@register_task("json_highest_lang_math_train")
class JSONMATHTrainTask(BaseMATHTrainTask):
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
    preprocessing=highest_language_content

@register_task("math_train_problem")
class ProblemMATHTrainTask(BaseMATHTrainTask):
    input_text=lambda x: x["problem"]
    evaluation=None

@register_task("math_train_solution")
class OpenR1Math220KTask(BaseMATHTrainTask):
    input_text=lambda x: x["solution"]
    evaluation=None

# @register_task("math_test")
# class LangBootMATHTestTask(MATHTask):
#     test_split="test"
#     # postprocessor=get_boxed_answer
#     postprocessor=None
#     evaluation={"accuracy": math_eval_with_postprocessing}

# @register_task("math_test_problem")
# class OpenR1Math220KTask(LangBootMATHTrainTask):
#     input_text=lambda x: x["question"]
#     evaluation=None

# @register_task("math_test_solution")
# class OpenR1Math220KTask(LangBootMATHTrainTask):
#     input_text=lambda x: x["answer"]
#     evaluation=None

if __name__ == "__main__":
    pass
