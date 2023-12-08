from django.shortcuts import render
from django.conf import settings
import torch


def predict(request):
    if request.method == 'POST':
        text = request.POST.get('mytext')
        model = torch.jit.load(settings.MODEL_PATH)
        model.eval()
        inputs = torch.tensor([15, 24, 38 ])
        all_encoder_layers, pooled_output = model(inputs)
        # inputs = torch.tensor([text])
        # outputs = model(inputs)
        # prediction = outputs.argmax(dim=-1).item()
        print(f'predictions are {all_encoder_layers} and {pooled_output}')
        return render(request, 'predictions.html', {'prediction': ''})
    else:
        return render(request, 'predictions.html')
