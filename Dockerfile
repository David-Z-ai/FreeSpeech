FROM nvidia/cuda:12.4.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

RUN conda create -n f5tts python=3.10 -y

RUN conda run -n f5tts pip install torch==2.4.1+cu124 torchaudio==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124

RUN conda run -n f5tts pip install \
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
    flask

WORKDIR /app
COPY app.py .

EXPOSE 5000

CMD ["conda", "run", "--no-capture-output", "-n", "f5tts", "python", "app.py"]