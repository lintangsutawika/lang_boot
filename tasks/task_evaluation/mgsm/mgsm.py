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
from yeval.response.math_responses import get_boxed_answer

path = os.path.dirname(__file__)

class MGSMTask(YevalTask):
    data_path="juletxara/mgsm"
    input_text=lambda x: f"Question:\n{x['question']}\nAnswer:"
    output_text=lambda x: x["answer_number"]
    test_split="test"
    evaluation={"accuracy": math_eval}
    postprocessor=get_boxed_answer

@register_task("mgsm_eng")
class MGSM_EN_Task(MGSMTask):
    data_name="en"

@register_task("mgsm_zho")
class MGSM_ZH_Task(MGSMTask):
    data_name="zh"

@register_task("mgsm_jpn")
class MGSM_JA_Task(MGSMTask):
    data_name="ja"

@register_task("mgsm_fra")
class MGSM_FR_Task(MGSMTask):
    data_name="fr"

@register_task("mgsm_ind")
class MGSM_ID_Task(MGSMTask):
    data_path="json"
    data_name=None
    data_kwargs={"data_files": {"test": os.path.join(path, "mgsm_id.jsonl")}}

@register_task("mgsm_translate")
class MGSMTranslateTask(YevalTask):
    system_message="You are a helpful assistant that can translate from English to Indonesian. Start your translation after ####"
    user_message=lambda x: "English Query:\n\n"+f"```\n{x}\n```"+"Indonesian Translation:\n\n"
    sampling_args={"n": 4}