import os
import tempfile
import soundfile as sf
from flask import Flask, request, send_file, jsonify
from ruaccent import RUAccent
from cached_path import cached_path
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import DiT
from ru_normalizr import Normalizer
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Устройство: {DEVICE}")

WEIGHTS_PATH = 'hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base_v2/model_last_inference.safetensors'
VOCAB_PATH = 'hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base/vocab.txt'

vocoder = load_vocoder(device=DEVICE)
ckpt_path = str(cached_path(WEIGHTS_PATH))
vocab_path = str(cached_path(VOCAB_PATH))
model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
model_obj = load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)

accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)
normalizer = Normalizer()

def generate(text, out_path, ref_audio, ref_text=""):
    normalized_text = normalizer.normalize(text)
    print(f"Оригинал: {text}\nНормализовано: {normalized_text}")
    text_stress = accentizer.process_all(normalized_text) + ' '
    ref_file, ref_text_proc = preprocess_ref_audio_text(ref_audio, ref_text)
    wav, sr, _ = infer_process(
        ref_file, ref_text_proc, text_stress, model_obj, vocoder,
        cross_fade_duration=0.15, nfe_step=64, speed=0.75, device=DEVICE
    )
    sf.write(out_path, wav, sr)
    return normalized_text

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    if 'text' not in request.form:
        return jsonify({'error': 'No text'}), 400

    audio_file = request.files['audio']
    text = request.form['text']

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_ref:
        audio_file.save(tmp_ref.name)
        ref_audio_path = tmp_ref.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_out:
        out_path = tmp_out.name

    try:
        generate(text, out_path, ref_audio_path, ref_text="")
        return send_file(out_path, as_attachment=True, download_name='output.wav')
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(ref_audio_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)