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

@register_task("en_reason")
class ENTranslateTask(BaseLangTask):
    # user_message=lambda x: x+"\n\nlet's solve this in English and write the final answer in \\boxed{}."
    user_message=lambda x: x+"\nReason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="en")}

@register_task("id_reason")
class IDTranslateTask(BaseLangTask):
    # user_message=lambda x: x+"\n\nmari kita pecahkan dalam Bahasa Indonesia dan tuliskan jawaban akhir di dalam \\boxed{}."
    user_message=lambda x: x+"\nBerpikir langkah demi langkah dan tuliskan jawaban akhir di dalam \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="id")}

@register_task("ja_reason")
class JATranslateTask(BaseLangTask):
    user_message=lambda x: x+"\n段階的に推論し、最終的な答えを\\boxed{}内に記入してください。"
    evaluation={"lang": partial(lang_content, lang="ja")}

@register_task("bn_reason")
class BNTranslateTask(BaseLangTask):
    user_message=lambda x: x+"\nধাপে ধাপে যুক্তি দিন এবং আপনার চূড়ান্ত উত্তর \\boxed{} এর মধ্যে লিখুন।"
    evaluation={"lang": partial(lang_content, lang="bn")}

@register_task("de_reason")
class DETranslateTask(BaseLangTask):
    user_message=lambda x: x+"\nBegründen Sie dies Schritt für Schritt und geben Sie Ihre endgültige Antwort in \\boxed{} ein."
    evaluation={"lang": partial(lang_content, lang="de")}

@register_task("es_reason")
class ESTranslateTask(BaseLangTask):
    user_message=lambda x: x+"\nRazona paso a paso y coloca tu respuesta final dentro de \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="es")}

@register_task("fr_reason")
class FRTranslateTask(BaseLangTask):
    user_message=lambda x: x+"\nRaisonner étape par étape et mettre votre réponse finale dans \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="fr")}

@register_task("ru_reason")
class RUTranslateTask(BaseLangTask):
    user_message=lambda x: x+"\nРассудите шаг за шагом и поместите окончательный ответ в \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="ru")}

@register_task("sw_reason")
class SWTranslateTask(BaseLangTask):
    user_message=lambda x: x+"\nSababu hatua kwa hatua na uweke jibu lako la mwisho ndani ya \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="sw")}

@register_task("te_reason")
class TWTranslateTask(BaseLangTask):
    user_message=lambda x: x+"\nదశలవారీగా తర్కించండి మరియు మీ తుది సమాధానాన్ని \\బాక్స్డ్{} లో ఉంచండి."
    evaluation={"lang": partial(lang_content, lang="te")}

@register_task("th_reason")
class THTranslateTask(BaseLangTask):
    user_message=lambda x: x+"\nอธิบายเหตุผลทีละขั้นตอนและใส่คำตอบสุดท้ายของคุณไว้ใน \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="th")}

@register_task("zh_reason")
class JPNTranslateTask(BaseLangTask):
    user_message=lambda x: x+"\n逐步推理并将您的最终答案放在 \\boxed{} 内。"
    evaluation={"lang": partial(lang_content, lang="zh")}


# class BaseBoxTask(YevalTask):
#     postprocessor=get_boxed_answer
#     logging=log_logprob

# @register_task("eng_generate_traces")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nReason step by step and put your final answer within \\boxed{}."
#     postprocessor=None
#     evaluation={"lang": partial(lang_content, lang="en")}
#     sample_agg_fn={"lang": lambda x: x}

# @register_task("ind_generate_traces")
# class IndReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nBerpikir langkah demi langkah dan tuliskan jawaban akhir di dalam \\boxed{}."
#     postprocessor=None
#     evaluation={"lang": partial(lang_content, lang="id")}
#     sample_agg_fn={"lang": lambda x: x}

# @register_task("zho_generate_traces")
# class ZhoReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\n请一步一步推理，并把最终答案写在 \\boxed{} 中。"
#     postprocessor=None
#     evaluation={"lang": partial(lang_content, lang="zh")}
#     sample_agg_fn={"lang": lambda x: x}

# @register_task("jpn_generate_traces")
# class JpnReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\n段階的に推論し、最終的な答えを\\boxed{}内に記入してください。"
#     postprocessor=None
#     evaluation={"lang": partial(lang_content, lang="ja")}
#     sample_agg_fn={"lang": lambda x: x}

# # English
# @register_task("eng_reason_A_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="Reason step by step and put your final answer within \\boxed{}."

# @register_task("eng_reason_A_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nReason step by step and put your final answer within \\boxed{}."

# @register_task("eng_reason_B_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="Think about it step by step and give your answer at the end in \\boxed{}."

# @register_task("eng_reason_B_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nThink about it step by step and give your answer at the end in \\boxed{}."

# @register_task("eng_reason_C_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="First give step by step reasoning, then write the answer within \\boxed{}."

# @register_task("eng_reason_C_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nFirst give step by step reasoning, then write the answer within \\boxed{}."

# # Indonesian
# @register_task("eng_reason_in_ind_A_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="Reason step by step in Indonesian and put your final answer within \\boxed{}."

# @register_task("eng_reason_in_ind_A_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nReason step by step in Indonesian and put your final answer within \\boxed{}."

# @register_task("eng_reason_in_ind_B_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="Think about it step by step in Indonesian and give your answer at the end in \\boxed{}."

# @register_task("eng_reason_in_ind_B_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nThink about it step by step in Indonesian and give your answer at the end in \\boxed{}."

# @register_task("eng_reason_in_ind_C_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="First give step by step reasoning in Indonesian, then write the answer within \\boxed{}."

# @register_task("eng_reason_in_ind_C_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nFirst give step by step reasoning in Indonesian, then write the answer within \\boxed{}."

# @register_task("ind_reason_A_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     system_message="Kamu adalah asisten yang senang membantu."
#     user_message="Berpikir langkah demi langkah dan tuliskan jawaban akhir di dalam \\boxed{}."

# @register_task("ind_reason_A_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nBerpikir langkah demi langkah dan tuliskan jawaban akhir di dalam \\boxed{}."

# @register_task("ind_reason_B_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="Berpikirlah tentang ini langkah demi langkah dan berikan jawaban Anda di akhir dalam \\boxed{}."

# @register_task("ind_reason_B_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nBerpikirlah tentang ini langkah demi langkah dan berikan jawaban Anda di akhir dalam \\boxed{}."

# @register_task("ind_reason_C_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="Pertama-tama berikan penalaran langkah demi langkah, lalu tuliskan jawabannya di dalam \\boxed{}."

# @register_task("ind_reason_C_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nPertama-tama berikan penalaran langkah demi langkah, lalu tuliskan jawabannya di dalam \\boxed{}."

# # Chinese
# @register_task("eng_reason_in_zho_A_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="Reason step by step in Chinese and put your final answer within \\boxed{}."

# @register_task("eng_reason_in_zho_A_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nReason step by step in Chinese and put your final answer within \\boxed{}."

# @register_task("eng_reason_in_zho_B_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="Think about it step by step in Chinese and give your answer at the end in \\boxed{}."

# @register_task("eng_reason_in_zho_B_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nThink about it step by step in Chinese and give your answer at the end in \\boxed{}."

# @register_task("eng_reason_in_zho_C_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="First give step by step reasoning in Chinese, then write the answer within \\boxed{}."

# @register_task("eng_reason_in_zho_C_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nFirst give step by step reasoning in Chinese, then write the answer within \\boxed{}."

# @register_task("zho_reason_A_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="请一步一步推理，并把最终答案写在 \\boxed{} 中。"

# @register_task("zho_reason_A_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\n请一步一步推理，并把最终答案写在 \\boxed{} 中。"

# @register_task("zho_reason_B_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="请逐步思考，最后将答案写在 \\boxed{} 中。"

# @register_task("zho_reason_B_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\n请逐步思考，最后将答案写在 \\boxed{} 中。"

# @register_task("zho_reason_C_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="先进行逐步推理，然后把答案写在 \\boxed{} 中。"

# @register_task("zho_reason_C_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\n先进行逐步推理，然后把答案写在 \\boxed{} 中。"

# # @register_task("eng_reason_in_jpn_00_box")
# # class EngReasonIndBoxTask(BaseBoxTask):
# #     user_message="Reason step by step in Japanese and put your final answer within \\boxed{}."

# # @register_task("jpn_reason_00_box")
# # class EngReasonIndBoxTask(BaseBoxTask):
# #     user_message="段階的に理論を展開し、最終的な答えを \\boxed{} の中に入れてください。"

# # Japanese
# @register_task("eng_reason_in_jpn_A_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="Reason step by step in Japanese and put your final answer within \\boxed{}."

# @register_task("eng_reason_in_jpn_A_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nReason step by step in Japanese and put your final answer within \\boxed{}."

# @register_task("eng_reason_in_jpn_B_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="Think about it step by step in Japanese and give your answer at the end in \\boxed{}."

# @register_task("eng_reason_in_jpn_B_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nThink about it step by step in Japanese and give your answer at the end in \\boxed{}."

# @register_task("eng_reason_in_jpn_C_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="First give step by step reasoning in Japanese, then write the answer within \\boxed{}."

# @register_task("eng_reason_in_jpn_C_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nFirst give step by step reasoning in Japanese, then write the answer within \\boxed{}."

# @register_task("jpn_reason_A_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="段階的に推論し、最終的な答えを\\boxed{}内に記入してください。"

# @register_task("jpn_reason_A_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\n段階的に推論し、最終的な答えを\\boxed{}内に記入してください。"

# @register_task("jpn_reason_B_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="段階的に考えて、最後に答えを\\boxed{}内に示してください。"

# @register_task("jpn_reason_B_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\n段階的に考えて、最後に答えを\\boxed{}内に示してください。"

# @register_task("jpn_reason_C_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="まず段階的な推論を示し、その後答えを\\boxed{}内に記述してください。"

# @register_task("jpn_reason_C_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nまず段階的な推論を示し、その後答えを\\boxed{}内に記述してください。"

# # French
# @register_task("eng_reason_in_fra_A_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="Reason step by step in French and put your final answer within \\boxed{}."

# @register_task("eng_reason_in_fra_A_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nReason step by step in French and put your final answer within \\boxed{}."

# @register_task("eng_reason_in_fra_B_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="Think about it step by step in French and give your answer at the end in \\boxed{}."

# @register_task("eng_reason_in_fra_B_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nThink about it step by step in French and give your answer at the end in \\boxed{}."

# @register_task("eng_reason_in_fra_C_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="First give step by step reasoning in French, then write the answer within \\boxed{}."

# @register_task("eng_reason_in_fra_C_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nFirst give step by step reasoning in French, then write the answer within \\boxed{}."

# @register_task("fra_reason_A_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="Raisonnez étape par étape et mettez votre réponse finale dans \\boxed{}."

# @register_task("fra_reason_A_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nRaisonnez étape par étape et mettez votre réponse finale dans \\boxed{}."

# @register_task("fra_reason_B_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="Réfléchissez étape par étape et donnez votre réponse à la fin dans \\boxed{}."

# @register_task("fra_reason_B_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nRéfléchissez étape par étape et donnez votre réponse à la fin dans \\boxed{}."

# @register_task("fra_reason_C_box_before")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message="Donnez d'abord un raisonnement étape par étape, puis écrivez la réponse dans \\boxed{}."

# @register_task("fra_reason_C_box_after")
# class EngReasonBoxTask(BaseBoxTask):
#     user_message=lambda x: f"{x}"+"\nDonnez d'abord un raisonnement étape par étape, puis écrivez la réponse dans \\boxed{}."
