"""Use the unchanged MotionGPT evaluator to compare the exact saved pair of banks."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from video_to_3d_motion.motion_generation.runtime import load_config, norms, write_json
from video_to_3d_motion.motion_generation.evaluation import Evaluator, seed_everything

def main():
    output=ROOT/'results/motiongpt3_comparison'
    base=ROOT/'results/motiongpt_stage1/baseline_20261019_230154/rep_00'
    cfg=load_config(ROOT/'configs/motiongpt_stage1/config.yaml')
    rows=[json.loads(line) for line in (output/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    matched=[row for row in rows if row['group']=='baseline']
    if len(matched)!=320:
        raise ValueError('All 320 original conditions must finish first')
    evaluator=Evaluator(cfg)
    neutral=norms(cfg)[0][0]
    embeddings=[]
    for start in range(0,len(matched),32):
        group=matched[start:start+32]
        lengths=[row['reference_frames'] for row in group]
        raw=np.broadcast_to(neutral,(len(group),max(lengths),263)).copy()
        for position,row in enumerate(group):
            with np.load(output/row['file'],allow_pickle=False) as arrays:
                if len(arrays['features'])!=lengths[position]:
                    raise ValueError('MotionGPT3 output length differs from validation protocol')
                raw[position,:lengths[position]]=arrays['features']
        embeddings.append(evaluator.embed_motion(raw,lengths))
    with np.load(base/'embeddings.npz',allow_pickle=False) as arrays:
        original={key:arrays[key] for key in arrays.files}
    generated=np.concatenate(embeddings)
    np.savez_compressed(output/'embeddings.npz',text=original['text'],real=original['real'],generated=generated)
    reports={}
    for name, bank in [('MotionGPT',original['generated']),('MotionGPT3',generated)]:
        # Reset RNG so both models have identical retrieval groups and diversity pairs.
        seed_everything(1234)
        reports[name]=evaluator.summarize(original['text'],bank,original['real'],300)
    for key in reports['MotionGPT']:
        if key.startswith('GT_') and reports['MotionGPT'][key]!=reports['MotionGPT3'][key]:
            raise ValueError(f'The common ground-truth grouping is inconsistent: {key}')
    report={'status':'paired_subset_evaluated','samples':320,'repetitions':1,'seed':1234,
            'metrics':reports,'invalid_motiongpt3':sum(not row['valid_generation'] for row in matched),
            'note':'MotionGPT embeddings reused without regeneration; both summaries recomputed with identical random grouping and diversity pair RNG. Original historical retrieval/Diversity figures may differ due to grouping.',
            'reference_length_note':'MotionGPT3 uses saved GT lengths per official validation. MotionGPT uses min(generated token length, reference length); lengths are not artificially stretched.'}
    write_json(output/'paired_metrics.json',report)
    old_panel=json.loads((ROOT/'results/motiongpt_stage1/critic_panel_five.json').read_text(encoding='utf-8'))
    panel_rows=[]
    for old in old_panel['samples']:
        source=Path(old['sample_path']).resolve()
        matches=[row for row in rows if Path(row['reference_path']).resolve()==source]
        if len(matches)!=1:
            raise ValueError('Cannot uniquely match a held-out MotionCritic condition')
        panel_rows.append(matches[0])
    (output/'critic_panel_samples.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in panel_rows),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    main()
