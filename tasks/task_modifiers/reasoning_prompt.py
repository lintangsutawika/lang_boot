from yeval.response.math_responses import get_boxed_answer
from yeval.task import register_task, YevalTask

class BaseBoxTask(YevalTask):
    postprocessor=get_boxed_answer

@register_task("eng_reason_A_box_before")
class EngReasonBoxTask(BaseBoxTask):
    user_message="Reason step by step and put your final answer within \\boxed{}."

@register_task("eng_reason_A_box_after")
class EngReasonBoxTask(BaseBoxTask):
    user_message=lambda x: f"{x}"+"\nReason step by step and put your final answer within \\boxed{}."

@register_task("eng_reason_B_box_before")
class EngReasonBoxTask(BaseBoxTask):
    user_message="Think about it step by step and give your answer at the end in \\boxed{}."

@register_task("eng_reason_B_box_after")
class EngReasonBoxTask(BaseBoxTask):
    user_message=lambda x: f"{x}"+"\nThink about it step by step and give your answer at the end in \\boxed{}."

@register_task("eng_reason_C_box_before")
class EngReasonBoxTask(BaseBoxTask):
    user_message="First give step by step reasoning, then write the answer within \\boxed{}."

@register_task("eng_reason_C_box_after")
class EngReasonBoxTask(BaseBoxTask):
    user_message=lambda x: f"{x}"+"\nFirst give step by step reasoning, then write the answer within \\boxed{}."

@register_task("eng_reason_in_ind_A_box_before")
class EngReasonBoxTask(BaseBoxTask):
    user_message="Reason step by step in Indonesian and put your final answer within \\boxed{}."

@register_task("eng_reason_in_ind_A_box_after")
class EngReasonBoxTask(BaseBoxTask):
    user_message=lambda x: f"{x}"+"\nReason step by step in Indonesian and put your final answer within \\boxed{}."

@register_task("eng_reason_in_ind_B_box_before")
class EngReasonBoxTask(BaseBoxTask):
    user_message="Think about it step by step in Indonesian and give your answer at the end in \\boxed{}."

@register_task("eng_reason_in_ind_B_box_after")
class EngReasonBoxTask(BaseBoxTask):
    user_message=lambda x: f"{x}"+"\nThink about it step by step in Indonesian and give your answer at the end in \\boxed{}."

@register_task("eng_reason_in_ind_C_box_before")
class EngReasonBoxTask(BaseBoxTask):
    user_message="First give step by step reasoning in Indonesian, then write the answer within \\boxed{}."

@register_task("eng_reason_in_ind_C_box_after")
class EngReasonBoxTask(BaseBoxTask):
    user_message=lambda x: f"{x}"+"\nFirst give step by step reasoning in Indonesian, then write the answer within \\boxed{}."

@register_task("ind_reason_A_box_before")
class EngReasonBoxTask(BaseBoxTask):
    user_message="Berpikir langkah demi langkah dan tuliskan jawaban akhir di dalam \\boxed{}."

@register_task("ind_reason_A_box_after")
class EngReasonBoxTask(BaseBoxTask):
    user_message=lambda x: f"{x}"+"\nBerpikir langkah demi langkah dan tuliskan jawaban akhir di dalam \\boxed{}."

@register_task("ind_reason_B_box_before")
class EngReasonBoxTask(BaseBoxTask):
    user_message="Berpikirlah tentang ini langkah demi langkah dan berikan jawaban Anda di akhir dalam \\boxed{}."

@register_task("ind_reason_B_box_after")
class EngReasonBoxTask(BaseBoxTask):
    user_message=lambda x: f"{x}"+"\nBerpikirlah tentang ini langkah demi langkah dan berikan jawaban Anda di akhir dalam \\boxed{}."

@register_task("ind_reason_C_box_before")
class EngReasonBoxTask(BaseBoxTask):
    user_message="Pertama-tama berikan penalaran langkah demi langkah, lalu tuliskan jawabannya di dalam \\boxed{}."

@register_task("ind_reason_C_box_after")
class EngReasonBoxTask(BaseBoxTask):
    user_message=lambda x: f"{x}"+"\nPertama-tama berikan penalaran langkah demi langkah, lalu tuliskan jawabannya di dalam \\boxed{}."

@register_task("zho_reason_A_box_before")
class EngReasonBoxTask(BaseBoxTask):
    user_message="请一步一步推理，并把最终答案写在 \\boxed{} 中。"

@register_task("zho_reason_A_box_after")
class EngReasonBoxTask(BaseBoxTask):
    user_message=lambda x: f"{x}"+"\n请一步一步推理，并把最终答案写在 \\boxed{} 中。"

@register_task("zho_reason_B_box_before")
class EngReasonBoxTask(BaseBoxTask):
    user_message="请逐步思考，最后将答案写在 \\boxed{} 中。"

@register_task("zho_reason_B_box_after")
class EngReasonBoxTask(BaseBoxTask):
    user_message=lambda x: f"{x}"+"\n请逐步思考，最后将答案写在 \\boxed{} 中。"

@register_task("zho_reason_C_box_before")
class EngReasonBoxTask(BaseBoxTask):
    user_message="先进行逐步推理，然后把答案写在 \\boxed{} 中。"

@register_task("zho_reason_C_box_after")
class EngReasonBoxTask(BaseBoxTask):
    user_message=lambda x: f"{x}"+"\n先进行逐步推理，然后把答案写在 \\boxed{} 中。"

# @register_task("eng_reason_in_jpn_00_box")
# class EngReasonIndBoxTask(BaseBoxTask):
#     user_message="Reason step by step in Japanese and put your final answer within \\boxed{}."

# @register_task("jpn_reason_00_box")
# class EngReasonIndBoxTask(BaseBoxTask):
#     user_message="段階的に理論を展開し、最終的な答えを \\boxed{} の中に入れてください。"
