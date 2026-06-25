# virtualenv ./venv
# ./venv/bin/python3 -m pip install transformers torch matplotlib numpy

from transformers import GPT2Model, GPT2Tokenizer
import matplotlib.pyplot as plt
import numpy as np ;
import torch 


# load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True)

# get embedding matrix to embed words
W = model.wte.weight.detach().numpy() ;
print("Embedding matirx dims", W.shape) ;

# compute some single-word embeddings, check that they align semantically by taking scalar product
# emb_cat = W[tokenizer.convert_tokens_to_ids("cat")] ;

emb_cat = W[tokenizer.convert_tokens_to_ids("cat")]
emb_dog = W[tokenizer.convert_tokens_to_ids("dog")]
emb_car = W[tokenizer.convert_tokens_to_ids("car")]

print("cat · dog =", np.dot(emb_cat, emb_dog))
print("cat · car =", np.dot(emb_cat, emb_car))
# !!!!!!!!!!!1

# Tokenize some text
text = "The quick brown fox looks at the lazy dog"
inputs = tokenizer(text, return_tensors="pt")
token_ids = np.array(inputs["input_ids"][0]) ;
tokens = [tokenizer.decode([token_id]) for token_id in inputs["input_ids"][0]]
print(tokens)
print(token_ids)


# !!!!!!!!!!!!!1
# generate a single one-column query matrix W_Q with thw word "dog"

# generate a single one-column query matrix W_K with "lazy"

# look up "dog" in sentence by Q = X*W_Q

# look up "lazy" in sentence by K = X*W_K

# form A as softmax(Q*K^T) and visualize Q*K^T

# ----------------------

# Run forward pass
with torch.no_grad():
    outputs = model(**inputs)
    
# extract attention matrix A of decoder layer 0 and vis.
A = outputs.attentions[0].numpy()

# visualize all 12 attenion heads of A
# !!!!!!!!!!!!!1


# extract Q, K, V
# Get the hidden states from the first layer
hidden_states = outputs.last_hidden_state  # Shape: [batch_size, seq_len, embed_dim]
# Access the Conv1D layer responsible for QKV projection in Layer 0
attn_mats = model.h[0].attn.c_attn.weight ;
fused_bias = model.h[0].attn.c_attn.bias      # Shape: [2304]
hidden_size = model.config.hidden_size
q_weight, k_weight, v_weight = [x.detach().numpy() for x in torch.split(attn_mats, hidden_size, dim=1)] ;
q_bias, k_bias, v_bias = [x.detach().numpy() for x in torch.split(fused_bias, hidden_size, dim=0)] ;

# Print shapes to verify
print("Query Matrix Shape:", q_weight.shape)  # Expected: torch.Size([768, 768])
print("Key Matrix Shape:", k_weight.shape)    # Expected: torch.Size([768, 768])
print("Value Matrix Shape:", v_weight.shape)  # Expected: torch.Size([768, 768])






