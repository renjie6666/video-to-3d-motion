"""Generate MotionGPT3 on the exact saved MotionGPT test conditions (one output each)."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import types
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'work/motiongpt3_source'
ASSETS = ROOT / 'checkpoints/motiongpt3'
OUTPUT = ROOT / 'results/motiongpt3_comparison'
BASE = ROOT / 'results/motiongpt_stage1/baseline_20261019_230154/rep_00'
PIN = 'cf5bab7daa568ce472e7201cc2bf8b83d8c1b732'
os.environ.setdefault('HF_HOME', str(ASSETS / 'hf_cache'))
os.environ.setdefault('TRANSFORMERS_CACHE', str(ASSETS / 'hf_cache'))
sys.path.insert(0, str(SOURCE))

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()

def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)

class Generator3(nn.Module):
    def __init__(self):
        super().__init__()
        if subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip() != PIN:
            raise ValueError('MotionGPT3 source revision changed')
        # The VAE imports a utility below models; bypass the unused Lightning trainer initializer.
        from motGPT.archs.mld_vae import MldVae
        from motGPT.archs.motgpt_lm import MLM
        checkpoint = torch.load(ASSETS / 'motiongpt3.ckpt', map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        self.checkpoint_hparams = checkpoint.get('hyper_parameters', {})
        # Architecture comes from the official merged configs. Final strict loads below
        # are mandatory: the empty bootstrap contains no inference weights.
        cfg = OmegaConf.merge(OmegaConf.load(SOURCE / 'configs/default.yaml'),
                              OmegaConf.load(SOURCE / 'configs/test.yaml'))
        bootstrap = ASSETS / 'gpt2/model_state_dict.pth'
        if not bootstrap.exists():
            torch.save({}, bootstrap)
        ablation = cfg.ABLATION
        vae_cfg = OmegaConf.to_container(OmegaConf.load(SOURCE / 'configs/vae/mldvae.yaml'))['params']
        vae_cfg.update(ablation=ablation, nfeats=263, datatype='humanml3d')
        self.vae = MldVae(**vae_cfg)
        lm_cfg = OmegaConf.to_container(cfg.lm_ablation)
        self.guidance = lm_cfg.pop('model_guidance_scale')
        lm_cfg.pop('instruction_type')
        diffhead = OmegaConf.load(SOURCE / 'configs/lm/diffloss.yaml')
        diffhead.params.multi_hidden = bool(lm_cfg['multi_hidden'])
        # Configs use learned unconditional latents only when CFG was enabled in training.
        if 'lm.fake_latent' in state:
            lm_cfg['guidance_scale'] = max(3., lm_cfg['guidance_scale'])
        self.lm = MLM(model_path=str(ASSETS / 'gpt2'), model_type='gpt2', stage=cfg.TRAIN.STAGE,
                      vae_latent_channels=256, vae_latent_size=1, diffhead=diffhead,
                      **lm_cfg)
        loaded = set()
        for module, prefix in [(self.vae,'vae.'), (self.lm,'lm.')]:
            subset = {key[len(prefix):]: value for key,value in state.items() if key.startswith(prefix)}
            module.load_state_dict(subset, strict=True)
            loaded.update(prefix + key for key in subset)
        remaining = sorted(set(state) - loaded)
        if any(not key.startswith(('_losses.','metrics.','TM2TMetrics.','PredMetrics.')) for key in remaining):
            raise ValueError(f'Unexpected unconsumed checkpoint weights: {remaining[:20]}')
        self.load_report = {'strict_lm_and_vae': True, 'loaded_tensors': len(loaded),
                            'unused_evaluation_tensors': remaining, 'guidance_scale': self.guidance,
                            'diffusion_sampling_steps': 100,
                            'construction_bootstrap_empty_replaced_by_full_checkpoint': True}
        self.mean = np.load(ROOT / 'datasets/HumanML3D/Mean.npy')
        self.std = np.load(ROOT / 'datasets/HumanML3D/Std.npy')
        # HumanML feature conversion is shared with the already validated baseline.
        package = types.ModuleType('motiongpt3_humanml')
        package.__path__ = [str(SOURCE / 'motGPT/data/humanml')]
        sys.modules[package.__name__] = package
        self.motion = importlib.import_module('motiongpt3_humanml.scripts.motion_process')
        self.eval().requires_grad_(False).cuda()

    @torch.inference_mode()
    def generate(self, caption, length):
        outputs = self.lm.generate_conditional([caption], lengths=[length], stage='test')
        latents, mask = self.lm.sample_tokens(outputs, torch.device('cuda:0'), temperature=1.,
                                             cfg=self.guidance, vae_mean_std_inv=self.vae.mean_std_inv)
        latents = latents.reshape(1, self.vae.latent_size, -1).permute(1,0,2)
        normalized = self.vae.decode(latents, [length])[0].cpu().numpy()
        valid = not bool(mask[0]) and np.isfinite(normalized).all()
        if not valid:
            normalized = np.zeros((length,263), dtype=np.float32)
        raw = normalized * self.std + self.mean
        joints = self.motion.recover_from_ric(torch.from_numpy(raw),22).numpy()
        return {'features':raw,'joints':joints,'latent':latents.cpu().numpy(),
                'valid_generation':bool(valid)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (BASE/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    panel = json.loads((ROOT/'results/motiongpt_stage1/critic_panel_five.json').read_text(encoding='utf-8'))
    conditions=[{**row,'group':'baseline','reference_path':str((BASE/row['file']).resolve()),
                 'output_file':'baseline_'+row['file']} for row in rows]
    for row in panel['samples']:
        if row['source_group']=='independent_single':
            conditions.append({**{key:row[key] for key in ('index','motion_id','caption','reference_frames','fps')},
                               'group':'independent_single','reference_path':row['sample_path'],
                               'output_file':'single_'+row['file']})
    protocol={'source_commit':PIN,'checkpoint_sha256':digest(ASSETS/'motiongpt3.ckpt'),
              'gpt2_metadata_sha256':digest(ASSETS/'gpt2_metadata.json'),
              'baseline_manifest_sha256':digest(BASE/'samples.jsonl'),'seed':1234,
              'sampler':{'official_config':'configs/test.yaml','guidance_scale':7.5,'diffusion_steps':100},
              'samples':320,'independent_extra':1,'repetitions':1,'outputs_per_condition':1,
              'length_policy':'official validation: decode to saved reference length',
              'note':'Matched text and reference clips; MotionGPT3 uses reference length, MotionGPT generates token length then crops. Not a full-paper benchmark.'}
    protocol_path=OUTPUT/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:
        raise ValueError('Existing run protocol differs; use a separate output directory')
    save(protocol_path,protocol)
    save(OUTPUT/'conditions.json',conditions)
    seed(1234)
    generator=Generator3()
    save(OUTPUT/'load_report.json',generator.load_report)
    manifest=[]
    selected=conditions[:1] if args.smoke else conditions
    for position,row in enumerate(selected):
        path=OUTPUT/row['output_file']
        per_seed=1234+position
        if not path.exists():
            seed(per_seed)
            record=generator.generate(row['caption'],row['reference_frames'])
            with np.load(row['reference_path'],allow_pickle=False) as reference:
                np.savez_compressed(path,**record,reference_features=reference['reference_features'])
        with np.load(path,allow_pickle=False) as generated:
            valid=bool(generated['valid_generation'])
            frames=len(generated['features'])
        manifest.append({**row,'file':row['output_file'],'frames':frames,'seed':per_seed,'valid_generation':valid})
        print(f'MotionGPT3 {position+1}/{len(selected)}: {row["motion_id"]}, {frames} frames, valid={valid}',flush=True)
        (OUTPUT/('smoke_samples.jsonl' if args.smoke else 'samples.jsonl')).write_text(
            ''.join(json.dumps(item,ensure_ascii=False)+'\n' for item in manifest),encoding='utf-8')
    save(OUTPUT/('smoke_result.json' if args.smoke else 'generation_result.json'),
         {'status':'generated','count':len(manifest),'invalid':sum(not row['valid_generation'] for row in manifest),
          'python':sys.executable,'conda_environment':str(sys.prefix),'torch':torch.__version__,
          'gpu':torch.cuda.get_device_name(0),'load_report':generator.load_report})

if __name__=='__main__':
    main()
