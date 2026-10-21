"""Check the matched conditions, reference clips and generated skeleton consistency."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from video_to_3d_motion.motion_generation.runtime import load_config, official_modules, write_json

def main():
    directory=ROOT/'results/motiongpt3_comparison'
    original=ROOT/'results/motiongpt_stage1/baseline_20261019_230154/rep_00/samples.jsonl'
    rows=[json.loads(line) for line in (directory/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    baseline=[json.loads(line) for line in original.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==321 and len(baseline)==320
    motion=official_modules(load_config(ROOT/'configs/motiongpt_stage1/config.yaml'))
    torch.set_num_threads(1)
    largest=0.
    for position,row in enumerate(rows):
        if position<320:
            assert all(row[key]==baseline[position][key] for key in ('caption','motion_id','index','reference_frames'))
        with np.load(row['reference_path'],allow_pickle=False) as reference, np.load(directory/row['file'],allow_pickle=False) as generated:
            assert np.array_equal(reference['reference_features'],generated['reference_features'])
            assert len(generated['features'])==row['reference_frames']
            assert bool(generated['valid_generation'])==row['valid_generation']
            recovered=motion.recover_from_ric(torch.from_numpy(generated['features']),22).numpy()
            error=float(np.abs(recovered-generated['joints']).max())
            assert error<1e-6
            largest=max(largest,error)
    metrics=json.loads((directory/'paired_metrics.json').read_text(encoding='utf-8'))
    for key,value in metrics['metrics']['MotionGPT'].items():
        if key.startswith('GT_'):
            assert value==metrics['metrics']['MotionGPT3'][key]
    result={'status':'passed','matched_baseline_conditions':320,'identical_reference_clips':321,
            'skeletons_checked':321,'max_recovery_difference_m':largest,'common_gt_metrics_identical':True}
    write_json(directory/'integrity_check.json',result)
    print(json.dumps(result))

if __name__=='__main__':
    main()
