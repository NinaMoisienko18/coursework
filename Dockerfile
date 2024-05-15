FROM python:3.11

# Встановлюємо необхідні бібліотеки для роботи з OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Встановлюємо Tesseract
RUN apt-get install -y tesseract-ocr

# Створюємо та переходимо в робочу директорію
WORKDIR /app

# Копіюємо файли вашого додатку в контейнер
COPY . /app

# Встановлюємо необхідні Python залежності
RUN pip install -r requirements.txt

# Встановлюємо шлях до виконуваного файлу Tesseract
ENV TESSERACT_CMD=/usr/bin/tesseract

# Запускаємо ваш додаток при старті контейнера
CMD ["python", "streamlit_app.py"]
