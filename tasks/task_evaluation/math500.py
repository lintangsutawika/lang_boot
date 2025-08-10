import os
import re
from functools import partial

from yeval.task import register_task, YevalTask

from yeval.log.usage import log_token_usage
from yeval.response import (
    match_routing,
    preprocess_routing,
    postprocess_routing
    )

from yeval.metrics import math_eval
from yeval.log.usage import log_logprob
from yeval.response.math_responses import get_boxed_answer

from lang_boot.utils import math_eval_with_postprocessing

path = os.path.dirname(__file__)

@register_task("math500")
class Math500Task(YevalTask):
    system_message="Think about it step by step and give your answer at the end in \\boxed{}."
    data_path="HuggingFaceH4/MATH-500"
    input_text=lambda x: x["problem"]
    output_text=lambda x: x["answer"]
    test_split="test"
    evaluation={"accuracy": math_eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob
