from functools import partial
from langdetect import detect_langs

from yeval.response.math_responses import get_boxed_answer
from yeval.task import register_task, YevalTask
from yeval.log.usage import log_logprob

from lang_boot.utils import get_lang_score

def lang_content(x, y, lang="id"):
    _, lang_prob = get_lang_score(x, lang=lang)
    return lang_prob

class BaseLangTask(YevalTask):
    system_message="Think about it step by step and give your answer at the end in \\boxed{}."
    sample_agg_fn={"lang": lambda x: x}
    logging=log_logprob

@register_task("bn_measure")
class BNTranslateTask(BaseLangTask):
    evaluation={"lang": partial(lang_content, lang="bn")}

@register_task("de_measure")
class DETranslateTask(BaseLangTask):
    evaluation={"lang": partial(lang_content, lang="de")}

@register_task("en_measure")
class ENTranslateTask(BaseLangTask):
    evaluation={"lang": partial(lang_content, lang="en")}

@register_task("es_measure")
class ESTranslateTask(BaseLangTask):
    evaluation={"lang": partial(lang_content, lang="es")}

@register_task("fr_measure")
class FRTranslateTask(BaseLangTask):
    evaluation={"lang": partial(lang_content, lang="fr")}

@register_task("ja_measure")
class JATranslateTask(BaseLangTask):
    evaluation={"lang": partial(lang_content, lang="ja")}

@register_task("ru_measure")
class RUTranslateTask(BaseLangTask):
    evaluation={"lang": partial(lang_content, lang="ru")}

@register_task("sw_measure")
class SWTranslateTask(BaseLangTask):
    evaluation={"lang": partial(lang_content, lang="sw")}

@register_task("te_measure")
class TWTranslateTask(BaseLangTask):
    evaluation={"lang": partial(lang_content, lang="te")}

@register_task("th_measure")
class THTranslateTask(BaseLangTask):
    evaluation={"lang": partial(lang_content, lang="th")}

@register_task("zh_measure")
class JPNTranslateTask(BaseLangTask):
    evaluation={"lang": partial(lang_content, lang="zh")}

@register_task("id_measure")
class IDTranslateTask(BaseLangTask):
    evaluation={"lang": partial(lang_content, lang="id")}
