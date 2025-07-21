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

@register_task("mgsm_bn")
class MGSM_BN_Task(MGSMTask):
    data_name="bn"

@register_task("mgsm_de")
class MGSM_DE_Task(MGSMTask):
    data_name="de"

@register_task("mgsm_en")
class MGSM_EN_Task(MGSMTask):
    data_name="en"

@register_task("mgsm_es")
class MGSM_ES_Task(MGSMTask):
    data_name="es"

@register_task("mgsm_fr")
class MGSM_FR_Task(MGSMTask):
    data_name="fr"

@register_task("mgsm_ja")
class MGSM_JA_Task(MGSMTask):
    data_name="ja"

@register_task("mgsm_ru")
class MGSM_RU_Task(MGSMTask):
    data_name="ru"

@register_task("mgsm_sw")
class MGSM_SW_Task(MGSMTask):
    data_name="sw"

@register_task("mgsm_te")
class MGSM_TW_Task(MGSMTask):
    data_name="te"

@register_task("mgsm_th")
class MGSM_TH_Task(MGSMTask):
    data_name="th"

@register_task("mgsm_zh")
class MGSM_ZH_Task(MGSMTask):
    data_name="zh"

@register_task("mgsm_id")
class MGSM_ID_Task(MGSMTask):
    data_path="json"
    data_name=None
    data_kwargs={"data_files": {"test": os.path.join(path, "mgsm_id.jsonl")}}

@register_task("mgsm_translate")
class MGSMTranslateTask(YevalTask):
    system_message="You are a helpful assistant that can translate from English to Indonesian. Start your translation after ####"
    user_message=lambda x: "English Query:\n\n"+f"```\n{x}\n```"+"Indonesian Translation:\n\n"
    sampling_args={"n": 4}