import streamlit as st
from PIL import Image
import pandas as pd
from pathlib import Path
import pickle

# Подключаем необходимые библиотеки
import matplotlib.pyplot as plt # для визуализации данных с помощью графиков

import requests # это мощный инструмент для работы с HTTP-запросами
import pandas as pd # позволяет удобно работать с данными в формате таблиц
import seaborn as sns # готовые шаблоны для статистической визуализации
import numpy as np # для работы с массивами и математическими операциями

from PIL import Image # для работы с изображениями
from IPython import display #  для отображения изображений из файлов в Jupyter Notebook

from typing import List, Tuple, Dict, Union, Callable # для аннотирования возвращаемых типов, позволяет определять тип переменной
from tqdm.notebook import tqdm # предназначен для быстрого и расширяемого внедрения индикаторов выполнения (progressbar)
from pathlib import Path # представляют путь к файлу или каталогу в файловой системе вашего компьютера.

# import tarfile
import zipfile # можно создавать, считывать, записывать zip-файлы

import os # функции для работы с операционной системой

import torch # Используется для решения различных задач: компьютерное зрение, обработка естественного языка
import torchvision # состоит из популярных наборов данных, архитектур моделей и общих преобразований изображений для компьютерного зрения
from torchvision import transforms # позволяет вам поворачивать, масштабировать, наклонять или сдвигать элемент
from torchvision.transforms import v2 # работа с изображениями
from torchvision.io import read_image # cчитывает изображение JPEG, PNG или GIF в трехмерное изображение RGB или оттенки серого

from ultralytics import YOLO


# 58_Face_Age_Gender_Emotion_Recognition_Streamlit_ONNX.ipynb
# ====================== главная страница ============================

# параметры главной страницы
# https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config
st.set_page_config(
    layout='wide',
    initial_sidebar_state='auto',
    page_title='Diabetes app/Plant Detection',
    page_icon='🧊',
)


# ----------- функции -------------------------------------

# функция для загрузки Главной картинки на странице
# кэшируем иначе каждый раз будет загружаться заново
@st.cache_data
def load_image(image_path):
    image = Image.open(image_path)
    return image

# функция загрузки модели
# кэшируем иначе каждый раз будет загружаться заново
@st.cache_data
def load_model(model_path):
    # загрузка модели
    with open(model_path, 'rb') as f:
        model = YOLO(f)
        # model = pickle.load(f)
    return model


# ------------- загрузка картинки для страницы и модели ---------

# путь до картинки-закинуть в проект на Гитхаб (практика/ 16 Streamlit Gradio Deploy/1:14)
image_path = 'main_page_image.jpg'
image = load_image(image_path)

# путь до модели. Загружаю ONNX модель
# model_path = 'model.pkl'
# diabet_model = load_model(model_path)
# путь до модели. Загружаю ONNX модель
model_path = 'best.onnx'
onnx_model = YOLO(model_path)

# ================= Пути до картинок примеров и прочие пути ===========

#EXAMPLES_DIR: Path = Path('examples_media')
#OUTPUT_RESULTS_DIR: Path = Path('output_results')
EXAMPLES_DIR: Path = ('main_example_image.jpg')
#MAIN_EXAMPLE_VIDEO_PATH: Path = EXAMPLES_DIR/ 'example_video.mp4'

# ---------- отрисовка текста и картинки ------------------------
st.write(
    """
    ### Диагностика диабета/Детекция и обнаружение заболеваний и состояний здоровья растений
    Введите ваши данные и получите результат
    """
)

# отрисовка картинки на странице
st.image(image, width=600)


# ====================== боковое меню для ввода данных ===============

st.sidebar.header('Настройка гиперпараметров')

conf_threshold = st.sidebar.slider(
    "Порог уверенности для детекции",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.1,
    )

iou_threshold = st.sidebar.slider(
    "Коэффициент Жаккарда",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.1,
    )
st.sidebar.write('---')

if 'image_detection_count' in st.session_state:
    st.sidebar.write(f'Кол-во детекций: {st.session_state.image_detection_count}')

# ================== Счетчики =====================================

def detection_counter():
    if 'image_detection_count' not in st.session_state:
        st.session_state.image_detection_count = 0
    st.session_state.image_detection_count += 1

def download_counter():
    if 'image_download_count' not in st.session_state:
        st.session_state.image_download_count = 0
    st.session_state.image_download_count += 1

# ==================== Загрузка изображения и распознавание ============

st_image = st.file_uploader(label='Выберите изображение')
result_pil_image = None

if st_image:
    pil_image = Image.open(st_image)
    st.image(pil_image, width=400)

    # ================ Кнопка распознать ==================
    if st.button('Распознать', on_click=detection_counter):
        with st.spinner('Распознавание фото ...'):
            # result_pil_image = detector_model.detect_image(pil_image, prob_threshold)
            result_pil_image = model.predict(source=pil_image, conf=prob_threshold, iou=iou_threshold, verbose=True )
# предикт модели входных данных, на выходе 1 из 46 классов растений и возможные заболевания

if result_pil_image is not None:
    st.image(result_pil_image)

    image_name = f"{st_image.name.split('.')[0]}_detect.jpg"

    file_buffer = io.BytesIO()
    result_pil_image.save(file_buffer, format='JPG')
    image_bytes = file_buffer.getvalue()

    # ================ Кнопка скачать ==================
    st.download_button('Скачать изображение', \
        image_bytes, image_name, on_click=download_counter)

if 'image_download_count' in st.session_state and st_image:
    st.success(f'Изображение {st_image.name} сохранено')

# предикт модели входных данных, на выходе 1 из 46 классов растений (и возможные заболевания)