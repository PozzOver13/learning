import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


def get_batch(train_data, split):
    # generate a small batch of data of inputs x and targets y
    data = train_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        # B = batch (4)
        # T = time (8)
        # C = channel (65)
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # RK: reshaping the tensors in order to be compliant with the pytorch implementation of cross entropy
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


if __name__ == "__main__":
    # read it in to inspect it
    with open('.\data\input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(''.join(chars))
    print(f"vocabulary size: {vocab_size}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    print(encode("hii there"))
    print(decode(encode("hii there")))

    batch_size = 4  # how many independent sequences will we process in parallel?
    block_size = 8  # what is the maximum context length for predictions?

    data = torch.tensor(encode(text), dtype=torch.long)

    xb, yb = get_batch(data, 'train')
    m = BigramLanguageModel(vocab_size)
    logits, loss = m(xb, yb)
    print(logits.shape)
    print(loss)

    print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long),
                            max_new_tokens=100)[0].tolist()))
