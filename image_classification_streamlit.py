from transformers import ViTImageProcessor, ViTForImageClassification
import streamlit as st
from PIL import Image
import requests


@st.cache_data
def load_model():
    return ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')


@st.cache_data
def load_processor():
    return ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')


def get_image_link():
    return st.text_input("Введите ссылку на изображение для распознавания")


def load_image(url):
    img = Image.open(requests.get(url, stream=True).raw)
    st.image(img)
    return img


def image_classification(picture):
    inputs = processor(images=picture, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]


def show_results(results):
    st.write(results)


processor = load_processor()
model = load_model()

st.title('Модель для классификации изображений vit-base-patch16-224')

link = get_image_link()

result = st.button('Распознать изображение')
if result:
    try:
        loaded_image = load_image(link)
        with st.spinner('Идет обработка... Пожалуйста, подождите...'):
            result = image_classification(loaded_image)
        st.write('**Результаты распознавания:**')
        st.write(result)
        st.snow()
    except IOError:
        st.error('Не удалось найти изображение по указанной ссылке. Попробуйте снова!')
