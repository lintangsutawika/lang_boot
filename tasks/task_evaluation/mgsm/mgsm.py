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

path = os.path.dirname(__file__)

class MGSMTask(YevalTask):
    data_path="juletxara/mgsm"
    input_text=lambda x: f"Question:\n{x['question']}\nAnswer:"
    output_text=lambda x: x["answer_number"]
    test_split="test"
    evaluation={"accuracy": math_eval}

@register_task("mgsm_eng")
class MGSM_ENTask(MGSMTask):
    data_name="en"

@register_task("mgsm_zho")
class MGSM_ENTask(MGSMTask):
    data_name="zh"

@register_task("mgsm_jpn")
class MGSM_ENTask(MGSMTask):
    data_name="ja"

@register_task("mgsm_ind")
class MGSM_IDTask(MGSMTask):
    data_path="json"
    data_name=None
    data_kwargs={"data_files": {"test": os.path.join(path, "mgsm_id.jsonl")}}
