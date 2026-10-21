"""Consolidate the completed matched-model metrics and five perceptual scores."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from video_to_3d_motion.motion_generation.runtime import write_json

def main():
    output=ROOT/'results/motiongpt3_comparison'
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    old=read(ROOT/'results/motiongpt_stage1/critic_panel_five.json')
    summary=read(output/'critic_summary.json')
    scores=[json.loads(line) for line in (output/'critic/scores.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(scores)!=5 or summary['scored']!=5 or any(row['status']!='scored' for row in scores):
        raise ValueError('All five perceptual results must complete')
    protocol={key:value for key,value in summary['protocol'].items() if key!='sample_count'}
    if protocol!=old['protocol']:
        raise ValueError('Perceptual conversion protocols differ')
    if {row['motion_id'] for row in scores}!={row['motion_id'] for row in old['samples']}:
        raise ValueError('Perceptual test identities differ')
    paired=[]
    for original in old['samples']:
        newer=next(row for row in scores if row['motion_id']==original['motion_id'])
        if newer['caption']!=original['caption']:
            raise ValueError('Perceptual text condition changed')
        paired.append({'motion_id':original['motion_id'],'caption':original['caption'],
                       'MotionGPT':original['critic']['score'],'MotionGPT3':newer['score'],
                       'difference':newer['score']-original['critic']['score'],
                       'motiongpt_frames':original['frames'],'motiongpt3_frames':newer['resampled_frames'],
                       'motiongpt3_fit_rmse_mean_m':newer['fit_rmse_mean_m'],
                       'motiongpt3_fit_rmse_max_m':newer['fit_rmse_max_m']})
    values=[row['MotionGPT3'] for row in paired]
    mean=float(np.mean(values))
    if abs(mean-summary['motioncritic_mean'])>1e-12:
        raise ValueError('Perceptual summary does not match raw scores')
    generation=read(output/'generation_result.json')
    integrity=read(output/'integrity_check.json')
    if generation['count']!=321 or integrity['status']!='passed':
        raise ValueError('Generation/integrity checks are incomplete')
    result={'status':'matched_subset_complete','generated_conditions':321,'common_metric_conditions':320,
            'perceptual_conditions':5,'coverage':1.,'generation_environment':generation['conda_environment'],
            'protocol':read(output/'protocol.json'),'common_metrics':read(output/'paired_metrics.json')['metrics'],
            'perceptual_means':{'MotionGPT':old['mean'],'MotionGPT3':mean},
            'motiongpt3_perceptual_std':float(np.std(values)),'perceptual_pairs':paired,
            'scope':'One subset run. MotionGPT3 decodes to GT reference length; MotionGPT uses its generated length capped by reference length. Five perceptual scores are not the mean of all 320 conditions.'}
    write_json(output/'comparison_result.json',result)
    print(json.dumps({'status':result['status'],'perceptual_means':result['perceptual_means']}))

if __name__=='__main__':
    main()
