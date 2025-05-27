import re
from yeval.task import register_task, YevalTask

def input_text(x):
    text = "\n"+x["step"][0]["full_input"][0]["content"]
    text = re.sub("(\n((?!\n).)*?boxed{}.)", "", text)
    return text.strip()

@register_task("ind_translate")
class TranslateTask(YevalTask):
    system_message="You are a helpful assistant that can translate from English to Indonesian. Start your translation after ####"
    user_message=lambda x: "English Query:\n\n"+f"```\n{x}\n```"+"Indonesian Translation:\n\n"
    input_text=input_text
    output_text=lambda x: x["ground_truth"]
    postprocessor=lambda x: x.replace("####", "").strip()
    data_path="json"
    sampling_args={
        # "n": 4,
        "n": 1,
    }
    test_split="train"


@register_task("default")
class TranslateTask(YevalTask):
    input_text=lambda x: x["answer"][0]
    output_text=lambda x: x["ground_truth"]
    data_path="json"
    sampling_args={
        "n": 4,
    }
    test_split="train"
