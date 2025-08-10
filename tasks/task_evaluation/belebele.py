import os
import re
from functools import partial

from yeval.task import register_task, YevalTask
from yeval.log.usage import log_logprob
from yeval.response.math_responses import get_boxed_answer

path = os.path.dirname(__file__)

def input_text(x):
    return f"{x['flores_passage']}\n{x['question']}\n\nA) {x['mc_answer1']}\nB) {x['mc_answer2']}\nC) {x['mc_answer3']}\nD) {x['mc_answer4']}\n\nAnswer: "

def output_text(x):
    label = int(x["correct_answer_num"])
    letter = ["A", "B", "C", "D"][label - 1]
    answer = x[f"mc_answer{label}"]
    return f"{letter}::{answer}"

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

class BelebeleTask(YevalTask):
    system_message="Think about it step by step and give your answer at the end in \\boxed{}."
    data_path="facebook/belebele"
    input_text=input_text
    output_text=output_text
    test_split="test"
    evaluation={"accuracy": eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob

@register_task("belebele_bn")
class Belebele_BN_Task(BelebeleTask):
    data_name="ben_Beng"

@register_task("belebele_de")
class Belebele_DE_Task(BelebeleTask):
    data_name="deu_Latn"

@register_task("belebele_en")
class Belebele_EN_Task(BelebeleTask):
    data_name="eng_Latn"

@register_task("belebele_es")
class Belebele_ES_Task(BelebeleTask):
    data_name="spa_Latn"

@register_task("belebele_fr")
class Belebele_FR_Task(BelebeleTask):
    data_name="fra_Latn"

@register_task("belebele_ja")
class Belebele_JA_Task(BelebeleTask):
    data_name="jpn_Jpan"

@register_task("belebele_ru")
class Belebele_RU_Task(BelebeleTask):
    data_name="rus_Cyrl"

@register_task("belebele_sw")
class Belebele_SW_Task(BelebeleTask):
    data_name="swh_Latn"

@register_task("belebele_te")
class Belebele_TE_Task(BelebeleTask):
    data_name="tel_Telu"

@register_task("belebele_th")
class Belebele_TH_Task(BelebeleTask):
    data_name="tha_Thai"

@register_task("belebele_zh")
class Belebele_ZH_Task(BelebeleTask):
    data_name="zho_Hans"

@register_task("belebele_id")
class Belebele_ID_Task(BelebeleTask):
    data_name="ind_Latn"
