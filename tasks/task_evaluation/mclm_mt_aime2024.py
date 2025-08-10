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

class MT_AIME2024Task(YevalTask):
    system_message="Think about it step by step and give your answer at the end in \\boxed{}."
    data_path="amphora/MCLM"
    data_name="MT-AIME2024"
    output_text=lambda x: x["answer"]
    test_split="test"
    evaluation={"accuracy": math_eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob

@register_task("mt_aime2024_bn")
class MT_AIME2024_BN_Task(MT_AIME2024Task):
    input_text=lambda x: x["bn"]

@register_task("mt_aime2024_de")
class MT_AIME2024_DE_Task(MT_AIME2024Task):
    input_text=lambda x: x["de"]

@register_task("mt_aime2024_en")
class MT_AIME2024_EN_Task(MT_AIME2024Task):
    input_text=lambda x: x["en"]

@register_task("mt_aime2024_es")
class MT_AIME2024_ES_Task(MT_AIME2024Task):
    input_text=lambda x: x["es"]

@register_task("mt_aime2024_fr")
class MT_AIME2024_FR_Task(MT_AIME2024Task):
    input_text=lambda x: x["fr"]

@register_task("mt_aime2024_ja")
class MT_AIME2024_JA_Task(MT_AIME2024Task):
    input_text=lambda x: x["ja"]

@register_task("mt_aime2024_ru")
class MT_AIME2024_RU_Task(MT_AIME2024Task):
    input_text=lambda x: x["ru"]

@register_task("mt_aime2024_sw")
class MT_AIME2024_SW_Task(MT_AIME2024Task):
    input_text=lambda x: x["sw"]

@register_task("mt_aime2024_te")
class MT_AIME2024_TW_Task(MT_AIME2024Task):
    input_text=lambda x: x["te"]

@register_task("mt_aime2024_th")
class MT_AIME2024_TH_Task(MT_AIME2024Task):
    input_text=lambda x: x["th"]

@register_task("mt_aime2024_zh")
class MT_AIME2024_ZH_Task(MT_AIME2024Task):
    input_text=lambda x: x["zh-cn"]

@register_task("mt_aime2024_id")
class MT_AIME2024_ZH_Task(MT_AIME2024Task):
    input_text=lambda x: x["id"]
