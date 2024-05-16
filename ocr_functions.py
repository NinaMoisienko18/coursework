import pytesseract
import numpy as np
import fitz
import streamlit as st
import tempfile
import docx
import cv2

pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

def perform_ocr(image_path, user_input_format):
    orig = cv2.imread(image_path)
    image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    col1, col2, col3 = st.columns((1, 3, 1))
    with col2:
        st.image(image, caption="Завантажене зображення", use_column_width=True)

    st.markdown(
        "<h1 style='color: #000000; text-align: center; font-size: 20px''>Попередня обробка зображення ⬇️</h1>",
        unsafe_allow_html=True)

    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel_sharpening)
    image = sharpened_image.copy()
    config = None

    if user_input_format == "Чек":
        config = r'--psm 4 --oem 3 -c preserve_interword_spaces=1'
    elif user_input_format == "Зображення з простим текстом (без таблиць тощо)":
        config = r'--psm 3 --oem 1 -c preserve_interword_spaces=1'
    elif user_input_format == "PDF-документ":
        text = extract_text_from_pdf(image_path)
        return None, None, text

    image = cv2.normalize(image, np.zeros((image.shape[0], image.shape[1])), 0, 255, cv2.NORM_MINMAX)

    remove_noise_img = cv2.fastNlMeansDenoisingColored(image, None, 5, 10, 10, 25)

    st.markdown("")
    st.image(remove_noise_img, caption=f"Обробка зображення", use_column_width=True)

    text = pytesseract.image_to_string(cv2.cvtColor(remove_noise_img, cv2.COLOR_BGR2RGB),
                                       lang="ukr+eng", config=config)

    results = pytesseract.image_to_data(remove_noise_img, output_type=pytesseract.Output.DICT)
    boxes = [((results['left'][i], results['top'][i]),
              (results['left'][i] + results['width'][i], results['top'][i] + results['height'][i]))
             for i in range(len(results['text'])) if results['text'][i].strip() != '']

    return remove_noise_img, boxes, text

def extract_text_from_pdf(pdf_path, num_pages=None):
    text = ""
    pdf_document = fitz.open(pdf_path)
    num_pages = min(num_pages, len(pdf_document)) if num_pages is not None else len(pdf_document)
    for page_num in range(num_pages):
        page = pdf_document[page_num]
        text += page.get_text("text")  # Додано параметр "text" для підтримки різних мов
    return text

def display_ocr_image(img, boxes):
    img_np = np.array(img)  # Перетворюємо зображення у numpy array

    img_with_rectangles = img_np.copy()  # Використовуємо numpy array для роботи з OpenCV

    for box in boxes:
        cv2.rectangle(img_with_rectangles, box[0], box[1], (0, 255, 0), 1)
    st.markdown(" ")


    st.image(img_with_rectangles, use_column_width=True, caption=f"Виділення розпізнаних об'єктів в прямокутники")

def download_file(text):
    export_option = st.selectbox("Оберіть формат для завантаження ⬇️", ["", "txt", "doc"])

    if export_option:
        if export_option == "txt":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(text.encode())
                st.success("Текст успішно збережено у форматі TXT.")
                st.download_button(label="Завантажити TXT", data=text, file_name="extracted_text.txt")
        elif export_option == "doc":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
                doc = docx.Document()
                doc.add_paragraph(text)
                doc.save(f.name)
                st.success("Текст успішно збережено у форматі DOCX.")
                st.download_button(label="Завантажити DOCX", data=open(f.name, "rb").read(), file_name="extracted_text.docx")
