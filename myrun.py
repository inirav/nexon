from transformers import BertTokenizer, BertForMaskedLM
import torch


loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()

enc = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)

# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = "[MASK]"
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]

all_encoder_layers, pooled_output = loaded_model(*dummy_input)

decoder_model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Generate text using the decoder model
generated_text = decoder_model.generate(input_ids=all_encoder_layers[-1], attention_mask=segments_tensors)

# Decode the generated text
decoded_text = enc.decode(generated_text[0].tolist())

print(f"Generated text: {decoded_text}")

#_________________________________________________________________________________________________________________________
# # ids = enc.convert_tokens_to_ids(all_encoder_layers)

# for layer in all_encoder_layers:
#     decoded_text = enc.decode(layer[0])
#     print(f"Layer: {decoded_text}")

# softmax_output = F.softmax(pooled_output, dim=-1)

# predicted_index = torch.argmax(softmax_output, dim=-1)
# predicted_token = enc.convert_ids_to_tokens([predicted_index.item()])
# print(f"Predicted token: {predicted_token}")

# for layer, decoded_text in enumerate(predicted_token):
#     if layer == masked_index:
#         decoded_text = decoded_text[:masked_index] + [predicted_token] + decoded_text[masked_index+1:]
#     print(f"Layer {layer}: {decoded_text}")

