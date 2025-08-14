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

# from lang_boot.utils import math_eval_with_postprocessing

path = os.path.dirname(__file__)

def input_text(x):
    # return f"{x['question']}\nA){x['option_a']}\nB){x['option_b']}\nC){x['option_c']}\nD){x['option_d']}\n"
    return f"{x['question']}\n\nA) {x['option_a']}\nB) {x['option_b']}\nC) {x['option_c']}\nD) {x['option_d']}\n\n"

def output_text(x):
    label = x["answer"].lower()
    answer = x[f"option_{label}"]
    return f"{label}::{answer}"

def eval_with_postprocessing(x, y):
    gold_letter, gold_answer = y.split("::")

    ans_score = 0.0
    ans = get_boxed_answer(x).lower()
    if ans.lower() == gold_letter.lower():
        ans_score = 1.0
    elif ans.lower() == gold_answer.lower():
        ans_score = 1.0
    elif ans.lower()[:1] == gold_letter.lower():
        ans_score = 1.0
    return ans_score

class GlobalMMLULiteTask(YevalTask):
    system_message="Think about it step by step and give your answer at the end in \\boxed{}."
    data_path="CohereLabs/Global-MMLU-Lite"
    input_text=input_text
    output_text=output_text
    test_split="test"
    evaluation={"accuracy": eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob

@register_task("global_mmlu_bn")
class GlobalMMLU_BN_Task(GlobalMMLULiteTask):
    data_name="bn"

@register_task("global_mmlu_de")
class GlobalMMLU_DE_Task(GlobalMMLULiteTask):
    data_name="de"

@register_task("global_mmlu_en")
class GlobalMMLU_EN_Task(GlobalMMLULiteTask):
    data_name="en"

@register_task("global_mmlu_es")
class GlobalMMLU_ES_Task(GlobalMMLULiteTask):
    data_name="es"

@register_task("global_mmlu_fr")
class GlobalMMLU_FR_Task(GlobalMMLULiteTask):
    data_name="fr"

@register_task("global_mmlu_ja")
class GlobalMMLU_JA_Task(GlobalMMLULiteTask):
    data_name="ja"

@register_task("global_mmlu_ru")
class GlobalMMLU_RU_Task(GlobalMMLULiteTask):
    data_name="ru"

@register_task("global_mmlu_sw")
class GlobalMMLU_SW_Task(GlobalMMLULiteTask):
    data_name="sw"

@register_task("global_mmlu_te")
class GlobalMMLU_TE_Task(GlobalMMLULiteTask):
    data_name="te"

@register_task("global_mmlu_th")
class GlobalMMLU_TH_Task(GlobalMMLULiteTask):
    data_name="th"

@register_task("global_mmlu_zh")
class GlobalMMLU_ZH_Task(GlobalMMLULiteTask):
    data_name="zh"

@register_task("global_mmlu_id")
class GlobalMMLU_ID_Task(GlobalMMLULiteTask):
    data_name="id"
    