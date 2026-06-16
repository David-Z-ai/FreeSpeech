import os
import sys
import time
import traceback
import tempfile
import soundfile as sf
from flask import Flask, request, send_file, jsonify

print("[LOG] Начало загрузки модулей...")

try:
    from ruaccent import RUAccent
    from cached_path import cached_path
    from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
    from f5_tts.model import DiT
    from ru_normalizr import Normalizer
    import torch
    print("[LOG] Все модули импортированы успешно.")
except Exception as e:
    print(f"[ERROR] Ошибка импорта модулей: {e}")
    traceback.print_exc()
    sys.exit(1)

print("[LOG] Определение устройства...")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[LOG] Устройство: {DEVICE}")

if DEVICE == 'cuda':
    print(f"[LOG] CUDA доступна, версия: {torch.version.cuda}")
    print(f"[LOG] Количество GPU: {torch.cuda.device_count()}")
    print(f"[LOG] Текущий GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[LOG] CUDA НЕ доступна, работаем на CPU (медленно)")

WEIGHTS_PATH = 'hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base_v2/model_last_inference.safetensors'
VOCAB_PATH = 'hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base/vocab.txt'

print("[LOG] Загрузка вокодера...")
try:
    vocoder = load_vocoder(device=DEVICE)
    print("[LOG] Вокодер загружен.")
except Exception as e:
    print(f"[ERROR] Ошибка загрузки вокодера: {e}")
    traceback.print_exc()
    sys.exit(1)

print("[LOG] Скачивание и загрузка модели...")
try:
    ckpt_path = str(cached_path(WEIGHTS_PATH))
    print(f"[LOG] Путь к весам: {ckpt_path}")
    vocab_path = str(cached_path(VOCAB_PATH))
    print(f"[LOG] Путь к словарю: {vocab_path}")
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    model_obj = load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)
    print("[LOG] Модель загружена успешно.")
except Exception as e:
    print(f"[ERROR] Ошибка загрузки модели: {e}")
    traceback.print_exc()
    sys.exit(1)

print("[LOG] Инициализация акцентизатора...")
try:
    accentizer = RUAccent()
    accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)
    print("[LOG] Акцентизатор загружен.")
except Exception as e:
    print(f"[ERROR] Ошибка загрузки акцентизатора: {e}")
    traceback.print_exc()
    sys.exit(1)

print("[LOG] Инициализация нормализатора...")
try:
    normalizer = Normalizer()
    print("[LOG] Нормализатор загружен.")
except Exception as e:
    print(f"[ERROR] Ошибка загрузки нормализатора: {e}")
    traceback.print_exc()
    sys.exit(1)

print("[LOG] ВСЕ КОМПОНЕНТЫ ЗАГРУЖЕНЫ. ЗАПУСК FLASK...")

# ---------- Функция генерации ----------
def generate(text, out_path, ref_audio, ref_text=""):
    print(f"[LOG] Генерация для текста: {text[:50]}...")
    try:
        normalized_text = normalizer.normalize(text)
        print(f"[LOG] Нормализовано: {normalized_text[:50]}...")
        text_stress = accentizer.process_all(normalized_text) + ' '
        ref_file, ref_text_proc = preprocess_ref_audio_text(ref_audio, ref_text)
        wav, sr, _ = infer_process(
            ref_file, ref_text_proc, text_stress, model_obj, vocoder,
            cross_fade_duration=0.15, nfe_step=64, speed=0.75, device=DEVICE
        )
        sf.write(out_path, wav, sr)
        print(f"[LOG] Аудио сохранено: {out_path}")
        return normalized_text
    except Exception as e:
        print(f"[ERROR] Ошибка в generate: {e}")
        traceback.print_exc()
        raise

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    print("[LOG] Получен запрос /generate")
    if 'audio' not in request.files:
        print("[ERROR] Нет файла audio")
        return jsonify({'error': 'No audio file'}), 400
    if 'text' not in request.form:
        print("[ERROR] Нет текста")
        return jsonify({'error': 'No text'}), 400

    audio_file = request.files['audio']
    text = request.form['text']
    print(f"[LOG] Текст: {text[:50]}...")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_ref:
        audio_file.save(tmp_ref.name)
        ref_audio_path = tmp_ref.name
        print(f"[LOG] Референс сохранён: {ref_audio_path}")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_out:
        out_path = tmp_out.name
        print(f"[LOG] Выходной файл: {out_path}")

    try:
        generate(text, out_path, ref_audio_path, ref_text="")
        print("[LOG] Генерация успешна, отправка файла...")
        return send_file(out_path, as_attachment=True, download_name='output.wav')
    except Exception as e:
        print(f"[ERROR] Ошибка обработки запроса: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(ref_audio_path)
        print("[LOG] Временный файл референса удалён")

if __name__ == '__main__':
    print("[LOG] Запуск Flask на 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
