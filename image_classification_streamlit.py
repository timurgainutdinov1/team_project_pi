# Импорт библиотек
from transformers import ViTImageProcessor, ViTForImageClassification
import streamlit as st
from PIL import Image
import requests

# Создание декоратора @st.cache_data для функции load_model() чтобы закешировать результат работы и не вызывать каждый раз заново
@st.cache_data
def load_model():
    return ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
# load_model() загружает модель ViTForImageClassification 
# Создание декоратора  @st.cache_data для функции load_processor() чтобы закешировать результат работы и не вызывать каждый раз заново
@st.cache_data
def load_processor():
    return ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
# Load_processor() загружает процессор ViTImageProcessor

def get_image_link():
    # Вывод текстового поля куда можно ввести ссылку на изображение
    return st.text_input("Введите ссылку на изображение для распознавания")

# Открытие указанного изображения, создание объекта image
def load_image(url):
    img = Image.open(requests.get(url, stream=True).raw)
    st.image(img)
    return img

# Принятие объекта image
def image_classification(picture):
    # обработка изображения в вид, подходящий для модели
    inputs = processor(images=picture, return_tensors="pt")
    # передача изображения в модель
    outputs = model(**inputs)
    logits = outputs.logits
   # определение индекса класса с наибольшей вероятностью
    predicted_class_idx = logits.argmax(-1).item()
    # возврат метки класса
    return model.config.id2label[predicted_class_idx]

# вывод результатов распознавания 
def show_results(results):
    st.write(results)


processor = load_processor()
model = load_model()
# заголовок приложения
st.title('Модель для классификации изображений vit-base-patch16-224')
# функция для ввода ссылки на изображение
link = get_image_link()
# создание кнопки result для распознавания
result = st.button('Распознать изображение')
# если кнопка нажата:
if result:
    try:
        loaded_image = load_image(link)
        # вывод спиннера с сообщением о текущей обработке изображения
        with st.spinner('Идет обработка... Пожалуйста, подождите...'):
            # распознавание изображения
            result = image_classification(loaded_image)
            # результаты распознавания
        st.write('**Результаты распознавания:**')
        st.write(result)
        st.snow()
        # если произошла ошибка 
    except IOError:
        st.error('Не удалось найти изображение по указанной ссылке. Попробуйте снова!')
