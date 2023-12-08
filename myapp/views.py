from django.shortcuts import render
from django.conf import settings
import torch


def predict(request):
    if request.method == 'POST':
        text = request.POST.get('mytext')
        model = torch.jit.load(settings.MODEL_PATH)
        inputs = torch.tensor([text])
        outputs = model(inputs)
        prediction = outputs.argmax(dim=-1).item()
        print(f'predictions are {prediction}')
        return render(request, 'predictions.html', {'prediction': prediction})
    else:
        return render(request, 'predictions.html')
