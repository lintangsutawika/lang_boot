import re
from functools import partial
from yeval.log.usage import log_token_usage
from yeval.task import register_task, YevalTask

def exit_after_budget_exhausted(x, state, budget=1024):
    total_output_tokens = sum([step['log']['output_tokens'] for step in state['step']])
    if total_output_tokens >= budget:
        return True
    else:
        return False

@register_task("budget=512")
class Budget512Task(YevalTask):
    loop_exit=partial(exit_after_budget_exhausted, budget=512)
    logging=log_token_usage
    loop_max=25
    # sampling_args={
    #     "n":4
    # }

@register_task("budget=1024")
class Budget1024Task(Budget512Task):
    loop_exit=partial(exit_after_budget_exhausted, budget=1024)

@register_task("budget=2048")
class Budget1024Task(Budget512Task):
    loop_exit=partial(exit_after_budget_exhausted, budget=2048)

@register_task("budget=4096")
class Budget1024Task(Budget512Task):
    loop_exit=partial(exit_after_budget_exhausted, budget=4096)

@register_task("budget=8192")
class Budget1024Task(Budget512Task):
    loop_exit=partial(exit_after_budget_exhausted, budget=8192)
