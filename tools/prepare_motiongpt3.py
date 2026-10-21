"""Download public GPT2 metadata; full MotionGPT3 checkpoint supplies all weights."""
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / 'checkpoints/motiongpt3/gpt2'
target.mkdir(parents=True, exist_ok=True)
api = requests.get('https://huggingface.co/api/models/openai-community/gpt2', timeout=60)
api.raise_for_status()
revision = api.json()['sha']
files = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']
records = []
for name in files:
    url = f'https://huggingface.co/openai-community/gpt2/resolve/{revision}/{name}'
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    (target / name).write_bytes(response.content)
    records.append({'file': name, 'url': url, 'sha256': hashlib.sha256(response.content).hexdigest()})
    print(name, len(response.content), flush=True)
(target.parent / 'gpt2_metadata.json').write_text(json.dumps({'revision': revision, 'files': records}, indent=2), encoding='utf-8')
