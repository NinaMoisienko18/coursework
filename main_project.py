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
        st.markdown(
            "<h1 style='color: #000000; text-align: center; font-size: 20px''>Попередня обробка зображення ⬇️</h1>",
            unsafe_allow_html=True)

        # Викликаємо функцію set_image_dpi для зміни роздільної здатності зображення

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

        def normalization(image):
            norm_img = np.zeros((image.shape[0], image.shape[1]))
            return cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)

        col1, col2, col3 = st.columns((1, 3, 1))
        with col2:
            normalixation_img = normalization(image)
            st.image(normalixation_img, caption=f"1 Крок - Нормалізація зображення", use_column_width=True)

            text = pytesseract.image_to_string(cv2.cvtColor(normalixation_img, cv2.COLOR_BGR2RGB), lang="ukr+eng",
                                                config=define_format(user_input_format))

            results = pytesseract.image_to_data(normalixation_img, output_type=pytesseract.Output.DICT)
            boxes = [((results['left'][i], results['top'][i]),
                      (results['left'][i] + results['width'][i], results['top'][i] + results['height'][i]))
                     for i in range(len(results['text'])) if int(results['conf'][i]) > 0]

            display_ocr_image(normalixation_img, boxes, text)

        def remove_noise(image):
            try:
                remove_noise_img = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
                return remove_noise_img
            except Exception as e:
                return f"Помилка видалення шумів: {str(e)}", None, None

        col1, col2, col3 = st.columns((1, 3, 1))
        with col2:
            remove_noise_img = remove_noise(image)
            st.image(remove_noise_img, caption=f"2 Крок - Видалення шумів", use_column_width=True)

            text = pytesseract.image_to_string(cv2.cvtColor(remove_noise_img, cv2.COLOR_BGR2RGB), lang="ukr+eng",
                                               config=define_format(user_input_format))

            results = pytesseract.image_to_data(remove_noise_img, output_type=pytesseract.Output.DICT)
            boxes = [((results['left'][i], results['top'][i]),
                      (results['left'][i] + results['width'][i], results['top'][i] + results['height'][i]))
                     for i in range(len(results['text'])) if int(results['conf'][i]) > 0]

            display_ocr_image(remove_noise_img, boxes, text)
            # return remove_noise_img, boxes, text

        def get_grayscale(image):
            try:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                denoised_image = cv2.medianBlur(gray_image, 5)

                return denoised_image
            except Exception as e:
                return f"Помилка конвертації в відтінки сірого: {str(e)}", None, None

        col1, col2, col3 = st.columns((1, 3, 1))
        with col2:
            gray_img = get_grayscale(image)
            st.image(gray_img, caption=f"Перетворення в відтінки сірого", use_column_width=True)

            text = pytesseract.image_to_string(gray_img, lang="ukr+eng",
                                               config=define_format(user_input_format))

            results = pytesseract.image_to_data(gray_img, output_type=pytesseract.Output.DICT)
            boxes = [((results['left'][i], results['top'][i]),
                      (results['left'][i] + results['width'][i], results['top'][i] + results['height'][i]))
                     for i in range(len(results['text'])) if int(results['conf'][i]) > 0]

            display_ocr_image(gray_img, boxes, text)

        def thresholding(image):
            try:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                denoised_image = cv2.medianBlur(gray_image, 5)
                _, thresholded_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return thresholded_image
            except Exception as e:
                return f"Помилка застосування порогової обробки: {str(e)}", None, None

        col1, col2, col3 = st.columns((1, 3, 1))
        with col2:
            thresholded_img = thresholding(image)
            st.image(thresholded_img, caption=f"Порогова обробка", use_column_width=True)

            text = pytesseract.image_to_string(thresholded_img, lang="ukr+eng",
                                               config=define_format(user_input_format))

            results = pytesseract.image_to_data(thresholded_img, output_type=pytesseract.Output.DICT)
            boxes = [((results['left'][i], results['top'][i]),
                      (results['left'][i] + results['width'][i], results['top'][i] + results['height'][i]))
                     for i in range(len(results['text'])) if int(results['conf'][i]) > 0]

            display_ocr_image(thresholded_img, boxes, text)
            return thresholded_img, boxes, text

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
    images = convert_from_path(pdf_path, first_page=page_number + 1, last_page=page_number + 1)
    return images[0]


def display_ocr_image(thresh, boxes, text):
    img_np = np.array(thresh)  # Перетворюємо зображення у numpy array

    img_with_rectangles = img_np.copy()  # Використовуємо numpy array для роботи з OpenCV

    for box in boxes:
        cv2.rectangle(img_with_rectangles, box[0], box[1], (0, 255, 255), 2)
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
        num_pages = st.number_input("Введіть кількість сторінок для обробки", min_value=1, max_value=100, value=1,
                                    step=1)

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(data.getvalue())
            temp_file_path = temp_file.name

        text = extract_text_from_pdf(temp_file_path, num_pages)

        st.markdown("<h4 style='color: #000000; text-align: center;'>Знайдений текст:</h4>", unsafe_allow_html=True)
        text_area = st.code(text)
        st.markdown("")
    else:
        img = Image.open(data)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
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
