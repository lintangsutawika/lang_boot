from yeval.task import register_task, YevalTask
from yeval.metrics import math_eval

def highest_loglikelihood(dataset):

    def get_most_likely_answer(example):
        example["input"] = example["answer"][example["logprob"].index(max(example["logprob"]))]
        example["output"] = example["ground_truth"]
        return example

    unused_columns = ['sample_id', 'total_step', 'task_step', 'step', 'current_loop', 'logprob', 'answer']

    dataset = dataset.map(get_most_likely_answer, remove_columns=unused_columns)
    return dataset

@register_task("local_json_task")
class TrainTask(YevalTask):
    data_path="json"
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
    test_split="train"
    evaluation=None
    preprocessing=highest_loglikelihood
