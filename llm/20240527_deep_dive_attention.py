# the following script is a deep dive into the attention mechanism of the model
# it is used to understand the attention mechanism of the model
# source: https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
# author: Sebastian Raschka

import torch
import torch.nn.functional as F

if __name__ == "__main__":
    sentence = 'Life is short, eat dessert first'

    # create a dictionary that maps words to integers
    dc = {s: i for i, s in enumerate(sorted(sentence.replace(',', '').split()))}

    print(dc)

    # convert the sentence to a list of integers
    sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
    print(sentence_int)

    # create an embedding layer
    torch.manual_seed(123)
    embed = torch.nn.Embedding(6, 16)
    embedded_sentence = embed(sentence_int).detach()

    print(embedded_sentence)
    print(embedded_sentence.shape)

    # create the query, key, and value matrices
    torch.manual_seed(123)

    d = embedded_sentence.shape[1]

    d_q, d_k, d_v = 24, 24, 28

    W_query = torch.nn.Parameter(torch.rand(d_q, d))
    W_key = torch.nn.Parameter(torch.rand(d_k, d))
    W_value = torch.nn.Parameter(torch.rand(d_v, d))

    # example using the second element of the sequence
    x_2 = embedded_sentence[1]
    query_2 = W_query.matmul(x_2)
    key_2 = W_key.matmul(x_2)
    value_2 = W_value.matmul(x_2)

    print(query_2.shape)
    print(key_2.shape)
    print(value_2.shape)

    # example using the entire sequence
    keys = W_key.matmul(embedded_sentence.T).T
    values = W_value.matmul(embedded_sentence.T).T

    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)

    # example using the fifth element of the sequence
    omega_24 = query_2.dot(keys[4])
    print(omega_24)

    # example using the entire sequence
    omega_2 = query_2.matmul(keys.T)
    print(omega_2)

    # calculate the attention weights normalized by the square root of d_k
    attention_weights_2 = F.softmax(omega_2 / d_k ** 0.5, dim=0)
    print(attention_weights_2)

    # calculate the context vector
    context_vector_2 = attention_weights_2.matmul(values)

    print(context_vector_2.shape)
    print(context_vector_2)

    # multi-head attention
    h = 3
    multihead_W_query = torch.nn.Parameter(torch.rand(h, d_q, d))
    multihead_W_key = torch.nn.Parameter(torch.rand(h, d_k, d))
    multihead_W_value = torch.nn.Parameter(torch.rand(h, d_v, d))

    # example using the second element of the sequence
    multihead_query_2 = multihead_W_query.matmul(x_2)
    print(multihead_query_2.shape)

    # get the multihead keys and values
    multihead_key_2 = multihead_W_key.matmul(x_2)
    multihead_value_2 = multihead_W_value.matmul(x_2)

    # expand the dimensions of the query, key, and value matrices
    stacked_inputs = embedded_sentence.T.repeat(3, 1, 1)
    print(stacked_inputs.shape)

    # calculate keys and values using batch matrix multiplication
    multihead_keys = torch.bmm(multihead_W_key, stacked_inputs)
    multihead_values = torch.bmm(multihead_W_value, stacked_inputs)
    print("multihead_keys.shape:", multihead_keys.shape)
    print("multihead_values.shape:", multihead_values.shape)

    # permute the dimensions of the multihead keys and values
    multihead_keys = multihead_keys.permute(0, 2, 1)
    multihead_values = multihead_values.permute(0, 2, 1)
    print("multihead_keys.shape:", multihead_keys.shape)
    print("multihead_values.shape:", multihead_values.shape)
