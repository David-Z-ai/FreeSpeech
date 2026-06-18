FROM nvidia/cuda:12.4.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

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

RUN pip install torch==2.4.1+cu124 torchaudio==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124

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
    cached-path \
    jupyterlab

WORKDIR /app
COPY app.py .

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token='freespeech'"]
