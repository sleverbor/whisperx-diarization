"""Build a separate target reference from selected media or existing samples.

This screens embeddings, not speech quality or identity. Review excluded samples
when source media is available. Neither evaluation clip is used for enrollment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def screen(samples, minimum_similarity, dimensions):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != dimensions:
        raise ValueError(f'Expected a matrix with {dimensions} columns')
    norms = np.linalg.norm(samples, axis=1)
    valid = np.isfinite(samples).all(axis=1) & (norms > 0)
    rows = np.flatnonzero(valid)
    if len(rows) < 3:
        raise ValueError('At least three valid target samples are required')
    unit = samples[valid] / norms[valid, None]
    affinity = unit @ unit.T
    np.fill_diagonal(affinity, np.nan)
    anchor = int(np.argmax(np.nanmedian(affinity, axis=1)))
    scores = unit @ unit[anchor]
    keep = scores >= minimum_similarity
    if keep.sum() < 3 or keep.sum() <= len(unit) / 2:
        raise ValueError('No consistent majority; inspect the enrollment media')
    accepted = unit[keep]
    centroid = accepted.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    return accepted, centroid, {
        'anchor_row': int(rows[anchor]),
        'accepted_rows': rows[keep].tolist(),
        'excluded_rows': sorted(set(range(len(samples))) - set(rows[keep].tolist())),
        'similarity_to_anchor': {str(int(row)): float(score) for row, score in zip(rows, scores)},
        'minimum_similarity': minimum_similarity,
    }


def collect_media(manifest_path, model_dir):
    """Enroll explicitly selected target windows; reject ambiguous face frames.

    Manifest paths are local videos, offsets/durations refer to those videos.
    Face ROIs are normalized [left, top, right, bottom], chosen by the operator.
    Speech detection screens silence, not other speakers: voice windows must
    already be target-only. Four-second windows never overlap.
    """
    import subprocess
    import cv2
    import torch
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    from speechbrain.inference.speaker import SpeakerRecognition
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    torch.set_num_threads(2); cv2.setNumThreads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    face_model = FaceAnalysis(name='buffalo_l', providers=providers, sess_options=options, allowed_modules=['detection','recognition'])
    face_model.prepare(ctx_id=0 if providers[0] == 'CUDAExecutionProvider' else -1, det_size=(640,640))
    voice_model = SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir=str(model_dir), run_opts={'device':device})
    manifest = json.loads(manifest_path.read_text())
    samples = {'voice':[], 'face':[]}; provenance = {'voice':[], 'face':[]}
    for source in manifest['sources']:
        path = Path(source['path']).expanduser()
        if not path.is_absolute(): path = manifest_path.parent / path
        offset = float(source.get('offset_seconds',0)); duration = float(source['duration_seconds'])
        if offset < 0 or duration <= 0: raise ValueError('Invalid source window')
        payload = subprocess.check_output(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-ss',str(offset),'-i',str(path),'-t',str(duration),'-vn','-ar','16000','-ac','1','-f','f32le','pipe:1'])
        audio = np.frombuffer(payload,dtype='<f4').copy()
        speech = get_speech_timestamps(audio, vad_options=VadOptions(threshold=.5), sampling_rate=16000)
        voice_count=0
        for left in range(0,len(audio),64000):
            right=min(left+64000,len(audio)); chunk=audio[left:right]
            if len(chunk)<32000: continue
            coverage=sum(max(0,min(right,item['end'])-max(left,item['start'])) for item in speech)/len(chunk)
            rms=float(np.sqrt(np.mean(chunk**2)))
            if coverage<.6 or rms<.003 or float(np.mean(np.abs(chunk)>.99))>.01: continue
            with torch.no_grad(): vector=voice_model.encode_batch(torch.from_numpy(chunk).unsqueeze(0).to(device)).flatten().cpu().numpy()
            samples['voice'].append(vector); provenance['voice'].append({'source':source.get('url',str(path)), 'start':offset+left/16000, 'end':offset+right/16000, 'speech_fraction':coverage})
            voice_count+=1
        roi=source.get('face_roi',[0,0,1,1])
        if len(roi)!=4 or not (0<=roi[0]<roi[2]<=1 and 0<=roi[1]<roi[3]<=1): raise ValueError('Invalid normalized face ROI')
        cap=cv2.VideoCapture(str(path)); face_count=0
        try:
            for t in np.arange(offset,offset+duration,2.5):
                cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000);ok,image=cap.read()
                if not ok: continue
                h,w=image.shape[:2];crop=image[int(roi[1]*h):int(roi[3]*h),int(roi[0]*w):int(roi[2]*w)]
                faces=[f for f in face_model.get(crop) if f.det_score>=.7 and min(f.bbox[2]-f.bbox[0],f.bbox[3]-f.bbox[1])>=60]
                if len(faces)!=1: continue
                samples['face'].append(faces[0].embedding); provenance['face'].append({'source':source.get('url',str(path)), 'timestamp':float(t),'roi':roi});face_count+=1
        finally: cap.release()
        if voice_count<2 or face_count<2: raise ValueError(f'Source has too few usable target samples: {path}')
        print('Source accepted',path.name,'voice',voice_count,'face',face_count,flush=True)
    return {kind:np.stack(value) for kind,value in samples.items()}, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--voice-samples', type=Path)
    parser.add_argument('--face-samples', type=Path)
    parser.add_argument('--source-manifest',type=Path)
    parser.add_argument('--model-dir',type=Path,default=Path('pretrained_models/spkrec-ecapa-voxceleb'))
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--voice-minimum', type=float, default=0.45)
    parser.add_argument('--face-minimum', type=float, default=0.45)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a new candidate name')
    for cutoff in (args.voice_minimum, args.face_minimum):
        if not 0 <= cutoff < 1:
            parser.error('Similarity cutoffs must be between zero and one')
    if args.source_manifest:
        raw, provenance = collect_media(args.source_manifest,args.model_dir)
    elif args.voice_samples and args.face_samples:
        raw = {'voice':np.load(args.voice_samples,allow_pickle=False),'face':np.load(args.face_samples,allow_pickle=False)}
        provenance = None
    else:
        parser.error('Supply --source-manifest or both sample matrices')
    results = {}
    arrays = {}
    for kind, path, threshold, dimensions in (
        ('voice', args.voice_samples, args.voice_minimum, 192),
        ('face', args.face_samples, args.face_minimum, 512),
    ):
        accepted, centroid, audit = screen(raw[kind], threshold, dimensions)
        if path is not None:
            audit.update(input_path=str(path.resolve()),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if provenance is not None:
            audit['sample_provenance']=provenance[kind]
        results[kind] = audit
        arrays[kind] = (accepted, centroid)
    # Never overwrite an existing reference, including a partially created one.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for kind, (samples, centroid) in arrays.items():
        np.save(args.output_dir / f'{kind}_raw_embeddings.npy',raw[kind])
        np.save(args.output_dir / f'{kind}_embeddings.npy', samples)
        np.save(args.output_dir / f'{kind}_embedding.npy', centroid)
    results['method'] = 'majority-medoid consistency screening; unverified candidate'
    results['evaluation_audio_used_for_enrollment'] = False
    if args.source_manifest: results['source_manifest']=json.loads(args.source_manifest.read_text())
    (args.output_dir / 'reference.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({kind: {'accepted': len(results[kind]['accepted_rows']), 'excluded': results[kind]['excluded_rows']} for kind in arrays}, indent=2))


if __name__ == '__main__':
    main()
