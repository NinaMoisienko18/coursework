import streamlit as st
from PIL import Image
from ocr_functions import perform_ocr, extract_text_from_pdf, display_ocr_image, download_file
import tempfile
import base64
main_bg = "background.png"

st.set_page_config(page_title="Text Recognition App", page_icon="🧾")


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
                         ["Завантажити фото 💻", "Завантажити PDF-документ"], key="photo_selection")

user_input_format = None
data = None

if selection == "Завантажити фото 💻":
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
        download_file(text)
        st.markdown("")
    else:

        img = Image.open(data)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_file.write(data.getvalue())
            temp_file_path = temp_file.name

        img_ocr, boxes, text = perform_ocr(temp_file_path, user_input_format)
        if img is not None:
            display_ocr_image(img_ocr, boxes)

        if text:
            st.markdown("<h4 style='color: #000000; text-align: center;'>Знайдений текст:</h4>", unsafe_allow_html=True)
            text_area = st.code(text)
            st.markdown("")

        download_file(text)


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % get_base64(png_file)
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background(main_bg)
