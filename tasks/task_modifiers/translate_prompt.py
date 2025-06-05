import re

from functools import partial
from langdetect import detect_langs

from yeval.task import register_task, YevalTask

def input_text(x):
    text = "\n"+x["step"][0]["full_input"][0]["content"]
    text = re.sub("(\n((?!\n).)*?boxed{}.)", "", text)
    return text.strip()

def lang_content(x, y, lang="id"):
    lang_prob = 0.0
    try:
        langs = detect_langs(x)
        for l in langs:
            if l.lang == lang:
                lang_prob = l.prob
    except Exception as e:
        lang_prob = 0.0

    return lang_prob

def get_longest_response(x):
    longest = 0
    chosen = ""
    x = x.replace("#", "").strip()
    candidates = x.split("\n")
    for candidate in candidates:
        candidate = candidate.strip()
        if len(candidate) > longest:
            longest = len(candidate)
            chosen = candidate
    return chosen.strip()

@register_task("ind_translate")
class INDTranslateTask(YevalTask):
    system_message="You are a helpful assistant that can translate from English to Indonesian. Start your translation after ####"
    user_message=lambda x: f"English Query:\n\n```\n{x}\n```\n\nIndonesian Translation:\n\n"
    # postprocessor=lambda x: x.replace("####", "").strip().split("\n\n")[0].strip()
    postprocessor=get_longest_response
    sampling_args={
        "extra_body":{"guided_regex": "####(.*)"},
        "n": 4,
        }
    evaluation={"lang": partial(lang_content, lang="id")}
    sample_agg_fn={"lang": lambda x: x}

@register_task("jpn_translate")
class JPNTranslateTask(INDTranslateTask):
    system_message="You are a helpful assistant that can translate from English to Japanese. Start your translation after ####"
    user_message=lambda x: "English Query:\n\n"+f"```\n{x}\n```"+"Japanese Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="ja")}

@register_task("zho_translate")
class JPNTranslateTask(INDTranslateTask):
    system_message="You are a helpful assistant that can translate from English to Chinese. Start your translation after ####"
    user_message=lambda x: "English Query:\n\n"+f"```\n{x}\n```"+"Chinese Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="zh")}

@register_task("fra_translate")
class JPNTranslateTask(INDTranslateTask):
    system_message="You are a helpful assistant that can translate from English to French. Start your translation after ####"
    user_message=lambda x: "English Query:\n\n"+f"```\n{x}\n```"+"French Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="fr")}

@register_task("default")
class TranslateTask(YevalTask):
    input_text=lambda x: x["answer"][0]
    output_text=lambda x: x["ground_truth"]
    data_path="json"
    sampling_args={
        "n": 4,
    }
    test_split="train"
