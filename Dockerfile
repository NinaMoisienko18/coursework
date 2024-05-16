FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get -y install tesseract-ocr \
    && apt-get install -y python3 python3-distutils python3-pip \
    && cd /usr/local/bin \
    && if [ ! -e python ]; then ln -s /usr/bin/python3 python; fi \
    && pip3 --no-cache-dir install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

RUN apt update \
  && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install pytesseract
RUN pip3 install opencv-python
RUN pip3 install pillow

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "streamlit_app.py"]


