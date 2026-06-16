FROM nvidia/cuda:12.4.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip

# Устанавливаем PyTorch с CUDA 12.4
RUN pip install torch==2.4.1+cu124 torchaudio==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Устанавливаем все остальные пакеты
RUN pip install \
    f5-tts \
    soundfile \
    ruaccent \
    ru-normalizr \
    transliterate \
    sentencepiece \
    num2words \
    pymorphy2 \
    pymorphy2-dicts \
    ipykernel \
    transformers==4.46.3 \
    flask \
    cached-path

WORKDIR /app
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
