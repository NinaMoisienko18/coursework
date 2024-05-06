import streamlit as st
import cv2
from PIL import Image
import pytesseract
import base64
import tempfile
import numpy as np
import pandas as pd
import fitz  # Імпорт PyMuPDF
from pdf2image import convert_from_path


pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

def perform_ocr(image_path, user_input_format):
    try:

        orig = cv2.imread(image_path)
        image = orig.copy()

        def define_format(user_input_format):
            if user_input_format == "Чек":
                config_1 = r'--psm 4 --oem 3'
                return config_1

            elif user_input_format == "PDF-документ":
                text = extract_text_from_pdf(image_path)
                return None, None, text

            elif user_input_format == "Зображення з простим текстом (без таблиць тощо)":
                config_1 = r'--psm 3 --oem 3'
                return config_1


        def get_grayscale(image):
            bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        def thresholding(image):
            return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        def opening(image):
            kernel = np.ones((5, 5), np.uint8)
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        col1, col2 = st.columns(2)
        with col1:
            gray = get_grayscale(image)
            st.image(gray, caption=f"1. Зображення у відтінках сірого", use_column_width=True)

        with col2:
            thresh = thresholding(gray)
            st.image(thresh, caption=f"2. Порогове зображення після застосування фільтру", use_column_width=True)

        with col1:
            opened = opening(gray)
            st.image(opened, caption=f"3. Відкриття - після застосування морфологічного перетворення",
                     use_column_width=True)

        text5 = pytesseract.image_to_string(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB), lang="ukr+eng",
                                            config=define_format(user_input_format))

        results = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
        boxes = [((results['left'][i], results['top'][i]),
                  (results['left'][i] + results['width'][i], results['top'][i] + results['height'][i]))
                 for i in range(len(results['text'])) if int(results['conf'][i]) > 0]

        return thresh, boxes, text5
    except Exception as e:
        return f"Помилка розпізнавання тексту: {str(e)}", None, None

def extract_text_from_pdf(pdf_path, num_pages=None):
    text = ""
    pdf_document = fitz.open(pdf_path)
    num_pages = min(num_pages, len(pdf_document)) if num_pages is not None else len(pdf_document)
    for page_num in range(num_pages):
        page = pdf_document[page_num]
        text += page.get_text("text")  # Додано параметр "text" для підтримки різних мов
    return text

def convert_pdf_to_image(pdf_path, page_number=0):
    images = convert_from_path(pdf_path, first_page=page_number+1, last_page=page_number+1)
    return images[0]


def display_ocr_image(thresh, boxes, text):
    img_np = np.array(thresh)  # Перетворюємо зображення у numpy array
    col1, col2, col3 = st.columns((0.5, 2, 0.5))
    with col2:
        img_with_rectangles = img_np.copy()  # Використовуємо numpy array для роботи з OpenCV

        for box in boxes:
            cv2.rectangle(img_with_rectangles, box[0], box[1], (0, 255, 0), 2)

        st.markdown(" ")
        st.markdown(" ")

        st.image(img_with_rectangles, use_column_width=True)

main_bg = "background.png"

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % get_base64(png_file)
    st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("")
st.markdown(
    "<h1 style='color: #000000; text-align: center; font-size: 35px''>Розпізнання тексту на зображеннях та PDF-файлах [🧾]</h1>",
    unsafe_allow_html=True)
st.markdown("")
st.markdown(
    "<div style='text-align: justify; font-size: 18px'><b>📌Text Recognition App</b> - це експериментальний додаток,"
    " який створений з метою витягування тексту із зображень та PDF-файлів, значно зекономить час та зусилля при внесенні даних! </div>",
    unsafe_allow_html=True)
st.markdown("")

selection = st.selectbox("Оберіть зручний спосіб для завантаження фото у програму 👇",
                        ["Завантажити фото 💻", "Зробити фото 📸", "Завантажити PDF-документ"], key="photo_selection")

user_input_format = None
data = None
if selection == "Зробити фото 📸":
    data = st.camera_input("Наведи камеру на текстову частину")
elif selection == "Завантажити фото 💻":
    data = st.file_uploader("Завантажте фото", type=['png', 'jpg', 'jpeg'], key="photo_uploader")
    user_input_format = st.selectbox("Формат зображення",
                                     ["Чек", "Зображення з простим текстом (без таблиць тощо)"])

elif selection == "Завантажити PDF-документ":
    data = st.file_uploader("Завантажте файл", type=['pdf'], key="document_uploader")

if data:
    if selection == "Завантажити PDF-документ":
        num_pages = st.number_input("Введіть кількість сторінок для обробки", min_value=1, max_value=100, value=1, step=1)

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(data.getvalue())
            temp_file_path = temp_file.name


        text = extract_text_from_pdf(temp_file_path, num_pages)

        st.markdown("<h4 style='color: #000000; text-align: center;'>Знайдений текст:</h4>", unsafe_allow_html=True)
        text_area = st.code(text)
        st.markdown("")
    else:
        img = Image.open(data)
        with tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False) as temp_file:
            temp_file.write(data.getvalue())
            temp_file_path = temp_file.name

        img_ocr, boxes, text = perform_ocr(temp_file_path, user_input_format)

        if img is not None:
            display_ocr_image(img_ocr, boxes, text)

        if text:
            st.markdown("<h4 style='color: #000000; text-align: center;'>Знайдений текст:</h4>", unsafe_allow_html=True)
            text_area = st.code(text)
            st.markdown("")

        if st.button("Експорт у файл", help="Експорт тексту у вибраний формат файлу", key="export_button"):
            file_type = st.selectbox("Оберіть формат файлу", ["txt"], key="file_type")
            if file_type == "txt":
                with open("extracted_text.txt", "w") as f:
                    f.write(text)
                st.success("Текст успішно експортовано у форматі TXT.")

set_background(main_bg)