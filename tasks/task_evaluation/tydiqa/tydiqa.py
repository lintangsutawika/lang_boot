import os

from functools import partial
from yeval.task import register_task, YevalTask
from yeval.prompt import YevalPrompt, register_prompt

from evaluate import load

dir_path = os.path.dirname(os.path.realpath(__file__))

squad_v2_metric = load("squad_v2")

def f1_score(x, y):
    predictions = [{'prediction_text': x, 'id': '0000', 'no_answer_probability': 0.}]
    references = [{'answers': {'answer_start': [0], 'text': [y]}, 'id': '0000'}]
    return squad_v2_metric.compute(predictions=predictions, references=references)["f1"]

def exact_match_score(x, y):
    predictions = [{'prediction_text': x, 'id': '0000', 'no_answer_probability': 0.}]
    references = [{'answers': {'answer_start': [0], 'text': [y]}, 'id': '0000'}]
    return squad_v2_metric.compute(predictions=predictions, references=references)["exact"]

def filter_tydiqa(dataset, lang="indonesian"):
    def contains_answer(x):
        answerable = False
        for annotations in x['annotations']:
            if annotations['minimal_answer']['plaintext_end_byte'] != -1:
                answerable = True
                break
        x['answerable'] = answerable
        return x
    
    dataset = dataset.filter(lambda x: x["language"].startswith(lang))
    dataset = dataset.map(contains_answer)
    return dataset.filter(lambda x: x['answerable'] == True)

def tydiqa_output(x):
    for annotations in x['annotations']:
        if annotations['minimal_answer']['plaintext_end_byte'] != -1:
            end = annotations['minimal_answer']['plaintext_end_byte']
            start = annotations['minimal_answer']['plaintext_start_byte']
            return x['document_plaintext'][start:end]
    return "None"

from yeval.response import get_boxed_answer

@register_task(f"tydiqa_indonesian")
class TydiQAIndonesianTask(YevalTask):
    data_path="json"
    data_kwargs={
        "data_files": {"dev":os.path.join(dir_path, "tydiqa-v1.0-dev.jsonl")}
        }
    preprocessing=partial(filter_tydiqa, lang="indonesian")
    input_text=lambda x: f"{x['document_plaintext']}\n\n{x['question_text']}"
    output_text=tydiqa_output
    test_split="dev"
    evaluation={"f1": f1_score, "exact_match": exact_match_score}

@register_task(f"tydiqa_japanese")
class TydiQAJapaneseTask(TydiQAIndonesianTask):
    preprocessing=partial(filter_tydiqa, lang="japanese")

def filter_goldp(dataset, lang="indonesian"):
    return dataset.filter(lambda x: x["id"].startswith(lang))

@register_task(f"tydiqa_goldp_ind")
class TydiQAGoldPIndonesianTask(TydiQAIndonesianTask):
    data_kwargs={
        "data_files": {"dev":os.path.join(dir_path, "tydiqa-goldp-v1.1-dev.jsonl")}
        }
    preprocessing=lambda x: partial(filter_goldp, lang="indonesian")(x)
    input_text=lambda x: f"{x['context']}\n\n{x['question']}"
    output_text=lambda x: x['answers']

# @register_task(f"tydiqa_goldp_zho")
# class TydiQAGoldPChineseTask(TydiQAIndonesianTask):
#     preprocessing=partial(filter_goldp, lang="zh")


if __name__ == "__main__":

    import json
    import jsonlines
    import datasets 

    data_samples = []
    with open("tydiqa-goldp-v1.1-dev.json") as file:
        data = json.load(file)
        for line in data['data']:
            paragraph = line['paragraphs'][0]
            data_samples.append({
                "title": line['title'],
                "context": paragraph['context'],
                "id": paragraph['qas'][0]['id'],
                "question": paragraph['qas'][0]['question'],
                "answers": paragraph['qas'][0]['answers'][0]['text'],
            })

    with open("tydiqa-goldp-v1.1-dev.jsonl", "w") as file:
        writer = jsonlines.Writer(file)
        for sample in data_samples:
            writer.write(sample)
        writer.close()
            