"""Audit one-action cached outputs and one native S9 clip against original data."""
from dataclasses import replace
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pickle
import argparse
from tools.audit_posemamba_reproduction import load_labels, source_cases
from video_to_3d_motion.stage_output.archive import load_sequence,write_json,write_csv
from video_to_3d_motion.stage_output.calibration import project_camera,CAMERA_54138969
from video_to_3d_motion.stage_output.metrics import align_procrustes
from video_to_3d_motion.pose_adapter.normalization import normalize_h36m
from video_to_3d_motion.pose_adapter.windows import prepare_sequence_windows
from video_to_3d_motion.pose3d.config import load_pose3d_config
from video_to_3d_motion.pose3d.backend import PoseMambaBackend, file_sha256

class ArrayUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        permitted={('numpy.core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype'),('numpy.core.multiarray','scalar')}
        if (module,name) not in permitted:
            raise pickle.UnpicklingError(f'Unexpected class {module}.{name}')
        return super().find_class(module,name)


def audit(data_path):
    ROOT=Path(__file__).resolve().parents[1]/'results/reproduction_audit_2026-10-10'
    with data_path.open('rb') as stream:
        data=ArrayUnpickler(stream).load()
    pickle_sha256=file_sha256(data_path)
    report={'source':'original official MotionBERT pickle from PoseMamba README','mappings':[],'cached_runs':[],'temporal_control':[]}
    selected={}
    for name,source,destination in source_cases():
        sequence=load_sequence(destination);annotations=load_labels(destination,sequence)
        split='train' if name.startswith('S1') else 'test'
        source_id='s_01_act_02_cam_01' if split=='train' else 's_09_act_02_subact_01_ca_01'
        values=data[split];sources=np.asarray(values['source']);indices=np.flatnonzero(sources==source_id)
        if not len(indices): raise RuntimeError('Official source not found: '+source_id)
        target=np.asarray(values['joint3d_image'])[indices]
        projected=project_camera(annotations.joints_camera_mm,distorted=False)
        first_errors=np.linalg.norm(target[...,:2]-projected[0],axis=-1).mean(axis=-1)
        first=int(first_errors.argmin())
        local_ids=first+sequence.source_frame_ids-sequence.source_frame_ids[0]
        if local_ids[-1]>=len(indices): raise RuntimeError('Source mapping exceeds action')
        matched=indices[local_ids]
        xy_error=np.linalg.norm(np.asarray(values['joint3d_image'])[matched,:,:2]-projected,axis=-1)
        # Independent exports have small intrinsics/rounding differences. Require
        # every joint on every selected frame to match within one tenth of a pixel.
        if xy_error.max()>.1: raise RuntimeError(f'Official labels mismatch {name}: {xy_error.max()} px')
        # Verify full relative image-3D, including depth and joint index order.
        camera=annotations.joints_camera_mm;relative=camera-camera[:,:1]
        ratio=CAMERA_54138969['focal'][0]/camera[:,:1,2:]+1/2000
        reconstructed=np.concatenate((projected-projected[:,:1],relative[...,2:]*ratio),axis=-1)
        original=target[local_ids]-target[local_ids,:1]
        depth_error=float(np.abs(original-reconstructed).max())
        mapping=dict(case=name,source=source_id,split=split,original_frames=len(indices),first_index=first,matched_frames=len(matched),max_xy_error_px=float(xy_error.max()),max_image3d_relative_error=depth_error,native_index_stride=int(np.median(np.diff(local_ids))))
        print('MAPPING',json.dumps(mapping),flush=True);report['mappings'].append(mapping)
        has_factor='2.5d_factor' in values and 'joints_2.5d_image' in values
        if has_factor:
            factors=np.asarray(values['2.5d_factor'])[matched].reshape(-1)
            target25=np.asarray(values['joints_2.5d_image'])[matched].astype(np.float64);target25-=target25[:,:1]
            ideal=original*factors[:,None,None]
            mapping['original_factor_target_consistency_mm']=float(np.abs(ideal-target25).max())
            gt25mm=np.asarray(values['joints_2.5d_image'])[indices].astype(np.float64);gt25mm-=gt25mm[:,:1]
            np.savez_compressed(destination/'official_25d_labels.npz',source_frame_ids=sequence.source_frame_ids,factors=factors,target_25d_mm=target25,sequence=np.array(sequence.metadata.get('evaluation_sequence', annotations.provenance['sequence'])),original_pickle_sha256=np.array(pickle_sha256))
            def score(pred,targets,factor):
                prediction=pred.astype(np.float64)*500*factor[:,None,None]
                return dict(mpjpe_official_25d_mm=float(np.linalg.norm(prediction-targets,axis=-1).mean()),pa_mpjpe_official_25d_mm=float(np.linalg.norm(align_procrustes(prediction,targets)-targets,axis=-1).mean()))
            cached=dict(case=name,frames=len(matched),**score(sequence.P_3d,target25,factors));report['cached_runs'].append(cached);print('CACHED',json.dumps(cached),flush=True)
            if 'GT 2D' in name:
                altered=load_sequence(destination.parent/(destination.name+'_xy_passthrough'))
                report['cached_runs'].append(dict(case=name+' + XY passthrough (distorted H5 input)',frames=len(matched),**score(altered.P_3d,target25,factors)))
        selected[(split,'GT' if 'GT' in name else 'detected')]=(sequence,indices,local_ids,values,destination)

    # One 243-frame native S9 window, compared with sparse input on the same 49 observed frames.
    sequence,indices,local_ids,values,destination=selected[('test','GT')]
    config=load_pose3d_config(destination/'config_snapshot.yaml')
    # GT comparison run has input provenance rather than a replay path; use detected reference.
    detected_summary=json.loads((ROOT/'s9_detected_full/run_summary.json').read_text())
    windows,_=prepare_sequence_windows(detected_summary['input'],detected_summary['input_summary'],mode='full')
    template=windows[0];backend=PoseMambaBackend(config.model)
    start=int(local_ids[0]);dense=np.arange(start,start+243);common=np.arange(0,243,5);sparse_common=np.arange(len(common))
    factor=np.asarray(values['2.5d_factor'])[indices].reshape(-1)
    gt=np.asarray(values['joints_2.5d_image'])[indices].astype(np.float64);gt-=gt[:,:1]
    report['temporal_controls']=dict(common_frames=len(common),native_original_indices=(start+common).tolist(),native_stride=1,sparse_stride=5,source='real original data; no interpolation',single_window=True)
    for name,positions,kind in [('native50_GT',dense,'GT'),('sparse10_GT_pinhole',local_ids[:243],'GT'),('native50_SH',dense,'SH')]:
        raw=np.asarray(values['joint3d_image' if kind=='GT' else 'joint_2d'])[indices[positions],:,:2]
        normalized=np.stack([normalize_h36m(points,(1000,1002)) for points in raw])
        n=len(positions);times=np.arange(n,dtype=np.float64)/(50 if name.startswith('native') else 10)
        window=replace(template,keypoints_norm=normalized[None],source_frame_ids=positions.astype(np.int64)+1,timestamps=times,source_timestamps=times,timestamp_sources=('official_native_data',)*243)
        pred=backend.predict(window).pose_3d
        np.savez_compressed(ROOT/(name+'.npz'),prediction=pred,input=normalized,original_indices=positions)
        keep=common if name.startswith('native') else sparse_common
        for passthrough in ([False,True] if kind=='GT' else [False]):
            current=pred.copy()
            if passthrough: current[...,:2]=normalized-normalized[:,:1]
            evaluation=score(current[keep],gt[start+common],factor[start+common])
            report['temporal_control'].append(dict(case=name+('_official_XY' if passthrough else ''),frames=len(keep),**evaluation))
        if name=='native50_SH': report['native_SH_243_frame_result']=dict(frames=243,**score(pred,gt[dense],factor[dense]))
        print('NATIVE',report['temporal_control'][-1],flush=True)
    cached=sequence.P_3d[:len(common)]
    report['temporal_control'].append(dict(case='cached_sparse10_GT_distorted',frames=len(common),**score(cached,gt[start+common],factor[start+common])))
    report['scope']='single S9 Directions action/window plus one S1 action; official P1/P2 formulas, no full benchmark claim'
    write_json(ROOT/'original_data_audit.json',report);write_csv(ROOT/'original_data_cached.csv',report['cached_runs']);write_csv(ROOT/'original_temporal_control.csv',report['temporal_control'])

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("work/reproduction_reference/h36m_sh_conf_cam_source_final.pkl"))
    audit(parser.parse_args().data.resolve())
