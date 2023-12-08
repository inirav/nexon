from django.db import models

class Prediction(models.Model):
    text = models.TextField()
    prediction = models.CharField(max_length=255)
