# Import / Install libraries
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import utils

device = utils.device


def get_model_and_tokenizer():
    model_name = utils.settings['MODELS']['sentiment_analysis_model']
    sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    return sentiment_tokenizer, sentiment_model


activations = {}


def get_activations(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu().numpy()

    return hook


def get_score(outputs):
    label_probabilities = F.softmax(outputs[0], dim=1).detach().cpu().numpy()  # [positive, negative, neutral label]
    sentiment_scores = [x[0] - x[1] for x in label_probabilities]
    return sentiment_scores


def get_sentiment(text, sentiment_tokenizer, sentiment_model):
    inputs = sentiment_tokenizer([text], padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = sentiment_model(**inputs)
    sentiment_score = get_score(outputs)[0]
    sentiment_features = activations['feats'][0]
    return sentiment_score, sentiment_features


def get_sentiment_batch(batch, sentiment_tokenizer, sentiment_model):
    hook_handle = sentiment_model.bert.pooler.dense.register_forward_hook(get_activations('feats'))

    inputs = sentiment_tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = sentiment_model(**inputs)
    sentiment_scores = get_score(outputs)
    sentiment_features = activations['feats']
    return sentiment_scores, sentiment_features


def apply(data_generator, target_function):
    sentiment_tokenizer, sentiment_model = get_model_and_tokenizer()

    # print(sentiment_model.config)
    # print("Model's state_dict:")
    # for param_tensor in sentiment_model.state_dict():
    #    print(param_tensor, "\t", sentiment_model.state_dict()[param_tensor].size())
    # hook_handle = sentiment_model.bert.encoder.layer[11].output.LayerNorm.register_forward_hook(get_activations('feats'))

    #hook_handle = sentiment_model.bert.pooler.dense.register_forward_hook(get_activations('feats'))

    # Get sentiment scores and features
    for i, local_batch in enumerate(data_generator):
        sentiment_scores, sentiment_features = get_sentiment_batch(local_batch, sentiment_tokenizer, sentiment_model)
        target_function(i, sentiment_scores, sentiment_features)
        if i % 1000 == 0:
            print(f'Currently at iteration {i}')

    del sentiment_model, sentiment_tokenizer
