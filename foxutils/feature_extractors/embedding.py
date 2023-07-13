import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models
from sklearn.decomposition import PCA

from foxutils.utils import core_utils

device = core_utils.device


def get_model():
    model_name = core_utils.settings['MODELS']['sentence_transformer_model']
    embedding_model = SentenceTransformer(model_name).to(device)
    return embedding_model


def get_embedding(text, embedding_model):
    embeddings = embedding_model.encode([text])
    return embeddings[0]


def get_embedding_batch(batch, embedding_model):
    embeddings = embedding_model.encode(batch)
    return embeddings


def transform_and_normalize(vecs, kernel, bias):
    """
        Applying transformation then standardize
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return normalize(vecs)


def normalize(vecs):
    """
        Standardization
    """
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5


def compute_kernel_bias(vecs):
    """
    Calculate Kernel & Bias for the final transformation - y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s ** 0.5))
    W = np.linalg.inv(W.T)
    return W, -mu


def whiten_and_get_embedding_batch(batch, embedding_model, ndim=128):
    vecs = get_embedding_batch(batch, embedding_model)

    # Finding Kernel
    kernel, bias = compute_kernel_bias([vecs])
    kernel = kernel[:, :ndim]
    embeddings = np.vstack(vecs)

    # Sentence embeddings can be converted into an identity matrix
    # by utilizing the transformation matrix
    embeddings = transform_and_normalize(embeddings,
                                         kernel=kernel,
                                         bias=bias
                                         )

    return embeddings


def get_pca_reduced_model(embedding_model, pca_train_sentences, ndim=128):
    train_embeddings = embedding_model.encode(pca_train_sentences, convert_to_numpy=True)

    pca = PCA(n_components=ndim)
    pca.fit(train_embeddings)
    pca_comp = np.asarray(pca.components_)

    dense = models.Dense(in_features=embedding_model.get_sentence_embedding_dimension(), out_features=ndim, bias=False,
                         activation_function=torch.nn.Identity())
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    embedding_model.add_module('dense', dense)
    print(f"Updated model to produce {ndim} dimensions")
    return embedding_model


def apply(data_generator, target_function, has_whitening=False, ndim=128, has_pca=False, pca_training_sentences=None):
    embedding_model = get_model()
    if has_pca:
        embedding_model = get_pca_reduced_model(embedding_model, pca_training_sentences, ndim)

    for i, local_batch in enumerate(data_generator):
        if has_whitening:
            embeddings = whiten_and_get_embedding_batch(local_batch, embedding_model, ndim)
        else:
            embeddings = get_embedding_batch(local_batch, embedding_model)
        target_function(i, embeddings)
        if i % 1000 == 0:
            print(f'Currently at iteration {i}')

    del embedding_model
