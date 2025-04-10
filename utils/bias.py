import numpy as np
import torch
import json
import os


def scale_vector(v):
    mean = np.mean(v)
    v_minus_mean = v - mean
    v_minus_mean = np.tanh(1000 * v_minus_mean)
    return v_minus_mean

# 输出文本嵌入
def get_embedding(device,embedding_model,embedding_tokenizer,sentence):
    input_ids = embedding_tokenizer.encode(sentence, return_tensors="pt", max_length=512, truncation="longest_first")
    input_ids = input_ids.to(device)
    with torch.no_grad():
        output = embedding_model(input_ids)
        # print(output)
    return output[0][:, 0, :]

def get_bias(device,embedding_tokenizer,embedding_model,transform_model,context_sentence,mapping):
    context_embedding = get_embedding(device,embedding_model,embedding_tokenizer,context_sentence)
    with torch.no_grad():
        output = transform_model(context_embedding).cpu()[0].detach().numpy()
    similarity_array = scale_vector(output)[mapping]
    # similarity_array = scale_vector(output)
    return torch.from_numpy(-similarity_array)

def cosine_similarity(x, y):
    dot_product = torch.sum(x * y, dim=-1)
    norm_x = torch.norm(x, p=2, dim=-1)
    norm_y = torch.norm(y, p=2, dim=-1)
    return dot_product / (norm_x * norm_y)
