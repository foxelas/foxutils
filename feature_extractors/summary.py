from transformers import AutoTokenizer, PegasusForConditionalGeneration

import utils

device = utils.device

def get_model_and_tokenizer():
    # summary_tokenizer = AutoTokenizer.from_pretrained(utils.summary_model)
    # summary_model = AutoModelForSeq2SeqLM.from_pretrained(utils.summary_model).to(device)
    summary_tokenizer = AutoTokenizer.from_pretrained(utils.summary_model)
    summary_model = PegasusForConditionalGeneration.from_pretrained(utils.summary_model).to(device)
    return summary_tokenizer, summary_model


def get_summary(text,summary_tokenizer, summary_model):
    inputs = summary_tokenizer([text], truncation=True, padding="longest", return_tensors="pt").to(
        device)
    summary_ids = summary_model.generate(inputs["input_ids"])
    outputs = summary_tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return outputs[0]


def get_summary_batch(batch, summary_tokenizer, summary_model):
    inputs = summary_tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(
        device)
    summary_ids = summary_model.generate(inputs["input_ids"])
    outputs = summary_tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return outputs


def get_summary_2(text, summary_tokenizer, summary_model):
    batch = summary_tokenizer([text], truncation=True, padding="longest", return_tensors="pt").to(device)
    translated = summary_model.generate(**batch)
    outputs = summary_tokenizer.batch_decode(translated, skip_special_tokens=True)
    return outputs[0]


def apply(data_generator, target_function):
    summary_tokenizer, summary_model = get_model_and_tokenizer()

    for i, local_batch in enumerate(data_generator):
        summaries = get_summary_batch(local_batch, summary_tokenizer, summary_model)
        target_function(i, summaries)
        if i % 1000 == 0:
            print(f'Currently at iteration {i}')

    del summary_tokenizer, summary_model
