from transformers import AutoTokenizer
import torch

model = torch.jit.load("bert.pt")

text = "This movie was amazing!"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded_inputs = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt") # tokenizer(text, return_tensors="pt")
print(f'inputs {encoded_inputs}')

outputs = model(**encoded_inputs)

# Accessing logits for example:
logits = outputs["logits"]

# Convert logits to class predictions
predictions = logits.argmax(dim=-1)

# Accessing labels based on predictions
label = tokenizer.decode(predictions.item())

print(f"Predicted label: {label}")
