import re

from functools import partial
from langdetect import detect_langs

from yeval.task import register_task, YevalTask
from yeval.log.usage import log_logprob

from lang_boot.utils import get_lang_score

def lang_content(x, y, lang="id"):
    _, lang_prob = get_lang_score(x, lang=lang)
    return lang_prob

class BaseTranslateTask(YevalTask):
    postprocessor=lambda x: x.replace("```", "").strip()
    sampling_args={
        "n": 4,
        "temperature": 1.0,
        "logprobs": True,
        }
    sample_agg_fn={"lang": lambda x: x}
    logging=log_logprob

@register_task("bn_translate")
class BNTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Bengali. \
Respond directly after \"Bengali Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"Bengali Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="bn")}

@register_task("de_translate")
class DETranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to German. \
Respond directly after \"German Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"German Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="de")}

@register_task("en_translate")
class ENTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to English. \
Respond directly after \"English Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"English Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="en")}

@register_task("es_translate")
class ESTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Spanish. \
Respond directly after \"Spanish Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"Spanish Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="es")}

@register_task("fr_translate")
class FRTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to French. \
Respond directly after \"French Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"French Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="fr")}

@register_task("ja_translate")
class JATranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Japanese. \
Respond directly after \"Japanese Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"Japanese Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="ja")}

@register_task("ru_translate")
class RUTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Russian. \
Respond directly after \"Russian Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"Russian Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="ru")}

@register_task("sw_translate")
class SWTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Swahili. \
Respond directly after \"Swahili Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"Swahili Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="sw")}

@register_task("te_translate")
class TWTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Telugu. \
Respond directly after \"Telugu Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"Telugu Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="te")}

@register_task("th_translate")
class THTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Thai. \
Respond directly after \"Thai Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"Thai Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="th")}

@register_task("zh_translate")
class JPNTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Chinese. \
Respond directly after \"Chinese Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"Chinese Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="zh")}

@register_task("id_translate")
class IDTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Indonesian. \
Respond directly after \"Indonesian Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"Indonesian Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="id")}

@register_task("default")
class TranslateTask(YevalTask):
    input_text=lambda x: x["answer"][0]
    output_text=lambda x: x["ground_truth"]
    data_path="json"
    sampling_args={
        "n": 4,
    }
    test_split="train"
