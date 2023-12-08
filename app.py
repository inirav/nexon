import torch

loaded_model = torch.jit.load("traced_bert.pt")

# Perform inference with the loaded model
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = loaded_model(**inputs)

# Extract predictions from the output
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

print(predictions)