# team_project_pi

Команда:
Гайнутдинов Тимур Радикович (РИМ-130907)
Романова Виктория Борисовна (РИМ-130907)

Выбранная модель: vit-base-patch16-224

Описание модели

Модель служит для классификации изображений. 
Vision Transformer (ViT) по сути представляет собой BERT, но применяется к изображениям.
Модель ViT была предварительно обучена на ImageNet-21k , наборе данных, состоящем из 14 миллионов изображений и 21 тысяч классов, и доработана на ImageNet , наборе данных, состоящем из 1 миллиона изображений и 1 тысяч классов.

Использование модели на платформе Google Colab

1) Выполнить установку библиотеки "transformers":

!pip install transformers

2) Выполнить следующий код, присвоив переменной url значение - ссылку на анализируемое изображение.

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
