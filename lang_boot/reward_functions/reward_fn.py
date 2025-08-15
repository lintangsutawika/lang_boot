import os
import re
import random
from yeval.metrics import math_eval
from yeval.response.math_responses import get_boxed_answer
from lang_boot.utils import math_eval_with_postprocessing, get_lang_score

from lang_boot.reward_functions.repetition_penalty import repetition_penalty

from yeval.task import TASK_LIST
from yeval.utils import import_modules

path = os.path.dirname(__file__)
import_modules(os.path.join(path, "../../tasks/"))

# Make dict for all tasks,
eval_tasks = ["mgsm", "global_mmlu", "belebele", "mt_aime2024", "mt_math100"]

eval_fn = {
    "math500": TASK_LIST["math500"]().eval,
    **{task: TASK_LIST[f"{task}_en"]().eval for task in eval_tasks}
}

def compute_score(data_source, solution_str, ground_truth, extra_info=None, use_lang=False, use_penalty=False, use_random=False):

    task_eval = extra_info["task"].split("/")[0]
    if task_eval == "train_dataset":
        try:
            gold = get_boxed_answer(ground_truth)
            if gold == "None":
                gold = ground_truth
        except:
            gold = ground_truth

        try:
            ans = get_boxed_answer(solution_str)
        except:
            ans = solution_str

        if use_random:
            ans_score = random.random() >= 0.5
        else:
            ans_score = math_eval(ans, gold)
    else:
        ans_score = eval_fn[task_eval](solution_str, ground_truth)["accuracy"]

    reward = ans_score

    lang = extra_info["lang"]
    _, lang_score = get_lang_score(solution_str, lang=lang)
    if use_lang and (lang != "en"):
        reward += lang_score

    penalty = 0
    N_gram_sizes = [2, 3, 4, 5]
    for N in N_gram_sizes:
        penalty += repetition_penalty(solution_str, window_size=N)
    penalty /= len(N_gram_sizes)

    if use_penalty and (lang != "en"):
        reward -= penalty

    return {
        "data_source": extra_info["task"],
        "score": reward,
        "task_score": ans_score,
        "lang_score": lang_score,
        "repetition_penalty": penalty,
        "use_lang": use_lang,
        "use_penalty": use_penalty,
        "use_random": use_random,
        "gold": ground_truth,
    }

def compute_score_reward_acc(data_source, solution_str, ground_truth, extra_info):
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        use_lang=False,
        use_penalty=True,
    )

def compute_score_reward_acc_and_lang_fn(data_source, solution_str, ground_truth, extra_info):
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        use_lang=True,
        use_penalty=True,
    )

def compute_score_reward_rand(data_source, solution_str, ground_truth, extra_info):
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        use_lang=False,
        use_penalty=False,
        use_random=True,
    )
