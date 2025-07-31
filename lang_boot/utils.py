import re

from yeval.metrics import math_eval
from yeval.response.math_responses import get_boxed_answer

from lingua import Language, LanguageDetectorBuilder

detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

def get_last_digit_string(text):
    matches = re.findall(r'\d+', text)
    return matches[-1] if matches else None

def get_last_number_string(text, pos=-1):
    # Matches various number formats: 123, 123.45, .45, 123,456.78
    matches = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+', text)
    if matches:
        if len(matches) < abs(pos):
            return None
        else:
            return matches[pos] if matches else None
    else:
        return None

def math_eval_with_postprocessing(x, y):
    """
    Evaluates the math problem and returns the accuracy.
    """
    # x = get_boxed_answer(x)
    # x = clean(x)
    # x = get_last_digit_string(x)
    ans = get_last_number_string(x)
    
    score = math_eval(ans, y)
    if score == 0.0:
        # Check the second last digit
        ans = get_last_number_string(x, pos=-2)
        return math_eval(ans, y)
    
    return score

def get_lang_score(prediction, lang="id"):
    lang_prob = 0.0
    try:
        total_len = 0
        lang_dict = {}
        for detected_lang in detector.detect_multiple_languages_of(prediction):
            total_len += detected_lang.word_count
            lang_code = detected_lang.language.iso_code_639_1.name.lower()
            if lang_code not in lang_dict:
                lang_dict[lang_code] = detected_lang.word_count
            else:
                lang_dict[lang_code] += detected_lang.word_count
        if lang in lang_dict:
            lang_prob = lang_dict[lang]/total_len
        else:
            lang_prob = 0.0
    except Exception as e:
        print(f"Error detecting language: {e}")
        lang_prob = 0.0

    return prediction, lang_prob

def highest_loglikelihood(dataset):

    def get_most_likely_answer(example):
        example["input"] = example["answer"][example["logprob"].index(max(example["logprob"]))]
        example["output"] = example["ground_truth"]
        return example

    unused_columns = ['sample_id', 'total_step', 'task_step', 'step', 'current_loop', 'logprob', 'answer']

    dataset = dataset.map(get_most_likely_answer, remove_columns=unused_columns)
    return dataset

def highest_language_content(dataset):

    def get_most_likely_answer(example):
        example["input"] = example["answer"][example["lang"].index(max(example["lang"]))]
        example["output"] = example["ground_truth"]
        return example

    unused_columns = ['sample_id', 'total_step', 'task_step', 'step', 'current_loop', 'logprob', 'answer']

    dataset = dataset.map(get_most_likely_answer, remove_columns=unused_columns)
    return dataset

def math_eval_with_postprocessing(x, y):
    """
    Evaluates the math problem and returns the accuracy.
    """

    def clean(x):
        if x.startswith("\\$"):
            x.replace("\\$", "")
        return x

    x = get_boxed_answer(x)
    x = clean(x)
    
    return math_eval(x, y)