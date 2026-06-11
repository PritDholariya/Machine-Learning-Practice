import numpy as np


VOCAB = "abcdefghijklmnopqrstuvwxyz .'"

print(len(VOCAB))


char_to_idx = {
    ch: i
    for i, ch in enumerate(VOCAB)
}

idx_to_char = {
    i: ch
    for i, ch in enumerate(VOCAB)
}


print(char_to_idx['c'])