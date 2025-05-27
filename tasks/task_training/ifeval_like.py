import random
import ast
import os

from yeval.task import register_task, YevalTask

from yeval.task.ifeval.ifeval import IFEvalLikeTask
from yeval.task.ifeval.evaluation_lib import (
    test_instruction_following_strict,
    test_instruction_following_loose
    )

path = os.path.dirname(__file__)

LANGUAGE_CODE = {
    "ind": "Indonesian",
    "eng": "English",
}

def shuffle(dataset):
    shuffled_dataset = dataset.shuffle(seed=random.randint(1000,5000))
    return shuffled_dataset.flatten_indices()

def _test_instruction_following_loose(x, y):
    if isinstance(y['kwargs'], str):
        y["kwargs"] = [{k:v for k,v in _kwargs.items() if v is not None} for _kwargs in eval(y["kwargs"].replace("null", "None"))]
    return test_instruction_following_strict(x, y)

@register_task("ifeval_like_ind")
class IFEvalLikeTask(IFEvalLikeTask):
    data_path="argilla/ifeval-like-data"
    data_name="filtered"
    input_text=lambda x: x['prompt']+f"Your ENTIRE response should be in {LANGUAGE_CODE["ind"]}."
    output_text=lambda x: {k: x[k] for k in ["instruction_id_list", "kwargs"]}
    # preprocessing=shuffle
    test_split="train"
    evaluation={
        "score": _test_instruction_following_loose,
        }

# # Add to Keyword checker
# def check
#     doc = nlp("Kita mengharuskan semua menyayangi anjing")
#     [word.lemma for sent in doc.sentences for word in sent.words]
