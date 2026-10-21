"""Build an offline three-way motion viewer from real saved outputs only."""
import base64
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from video_to_3d_motion.motion_generation.runtime import load_config, official_modules

def main():
    output=ROOT/'results/motiongpt3_comparison'
    cfg=load_config(ROOT/'configs/motiongpt_stage1/config.yaml')
    motion=official_modules(cfg)
    torch.set_num_threads(1)
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (output/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    samples=[]
    for row in rows:
        with np.load(row['reference_path'],allow_pickle=False) as original, np.load(output/row['file'],allow_pickle=False) as newer:
            sequences={'motiongpt':original['joints'],'motiongpt3':newer['joints'],
                       'reference':motion.recover_from_ric(torch.from_numpy(original['reference_features'].astype(np.float32)),22).numpy()}
        for key,values in sequences.items():
            if values.ndim!=3 or values.shape[1:]!=(22,3) or not len(values) or not np.isfinite(values).all():
                raise ValueError(f'Invalid skeleton {row["motion_id"]}: {key}')
        samples.append({**row,**{key:value.astype(np.float64).round(4).tolist() for key,value in sequences.items()}})
    scores=output/'critic/scores.jsonl'
    payload={'samples':samples,'metrics':read(output/'paired_metrics.json'),'protocol':read(output/'protocol.json'),
             'panel':read(ROOT/'results/motiongpt_stage1/critic_panel_five.json'),
             'critic':[json.loads(line) for line in scores.read_text(encoding='utf-8').splitlines()] if scores.exists() else [],
             'critic_summary':read(output/'critic_summary.json') if (output/'critic_summary.json').exists() else None,
             'chains':[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]}
    encoded=base64.b64encode(gzip.compress(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(),mtime=0)).decode()
    template=(ROOT/'src/video_to_3d_motion/motion_generation/comparison.html').read_text(encoding='utf-8')
    target=output/'experiment_viewer.html'
    target.write_text(template.replace('__REPORT_GZIP_BASE64__',encoded),encoding='utf-8')
    print(json.dumps({'html':str(target),'conditions':len(samples),'bytes':target.stat().st_size}))

if __name__=='__main__':
    main()
