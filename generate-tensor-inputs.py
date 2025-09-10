from transformers import AutoTokenizer
from gptqmodel import GPTQModel
import torch
import torch.nn.functional as F
import numpy as np

quant_dir = "/workspace/hf_models"

# Sentences we want sentence embeddings for
sentences = ['Hello, world!']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(quant_dir)
model = GPTQModel.from_quantized(quant_dir)

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='np')

print(encoded_input)

np.save('input_ids', encoded_input['input_ids'])
# np.save('token_type_ids', encoded_input['token_type_ids'])
np.save('attention_mask', encoded_input['attention_mask'])


# Create your array
past_key = np.zeros((1, 8, 0, 128), dtype=np.float32)

# Save to .npy file
np.save('past_key_values.npy', past_key)