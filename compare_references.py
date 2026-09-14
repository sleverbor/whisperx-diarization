"""Controlled reference-only comparison using frozen upstream checkpoints.
Opening checkpoint: same source 0-30s, separately encoded Kaggle clip.
Second checkpoint: exact local 10:25-11:30 media digest.
Visual evidence is always recomputed for each reference.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--opening-checkpoints',type=Path,required=True)
    parser.add_argument('--second-checkpoints',type=Path,required=True)
    parser.add_argument('--second-video',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.project));os.environ['MPLBACKEND']='Agg'
    from dotenv import load_dotenv
    load_dotenv(args.project/'.env')
    import chainofrules as core
    core.torch.set_num_threads(2)
    core.cv2.setNumThreads(2)
    original_factory=core.FaceAnalysis
    options=core.ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    core.FaceAnalysis=lambda **kwargs: original_factory(sess_options=options,**kwargs)
    from cloud_runtime import StageCache
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.opening_checkpoints) as archive:
        prefix=next(n.rsplit('/',1)[0]+'/' for n in archive.namelist() if n.endswith('/manifest.json') and archive.read(n).find(b'bc5bef96708cfd')>=0)
        opening={Path(n).stem:json.loads(archive.read(n)) for n in archive.namelist() if n.startswith(prefix) and n.endswith('.json')}
    second={p.stem:json.loads(p.read_text()) for p in args.second_checkpoints.glob('*.json')}
    for stage in ('transcription','alignment'):
        if stage+'_vad' not in second and stage in second:
            second[stage+'_vad']=second[stage]
    for clip,video,frozen,offset in [('opening',args.project/'short.mp4',opening,0),('second',args.second_video,second,625)]:
        if clip=='second':
            from cloud_runtime import file_digest
            assert file_digest(video)==frozen['manifest']['inputs']['video'],'Second clip does not match checkpoint'
        for reference,prior_dir in [('original',args.project),('candidate',args.candidate)]:
            if (args.output_dir/f'{clip}_{reference}.json').exists(): continue
            class FrozenUpstream(StageCache):
                def __init__(self,*unused):
                    self.root=None
                def read(self,name):
                    if name in ('transcription_vad','alignment_vad','diarization') or name.startswith('voice_'):
                        return frozen.get(name)
                    if reference == 'original' and name.startswith('visual_'):
                        return frozen.get(name)
                    return None
                def get(self,name,compute):
                    saved=self.read(name)
                    if saved is None and name in ('transcription_vad','alignment_vad','diarization'):
                        raise RuntimeError('Missing frozen stage '+name)
                    return saved if saved is not None else compute()
            core.StageCache=FrozenUpstream
            output=args.output_dir/f'{clip}_{reference}.json'
            sys.argv=['chainofrules.py',str(video),'--voice-priors',str(prior_dir/'voice_embeddings.npy'),'--face-priors',str(prior_dir/'face_embeddings.npy'),'--output',str(output)]
            core.main()
            result=json.loads(output.read_text());result['source_offset_seconds']=offset
            result['experiment']='Reference-only comparison; frozen ASR, alignment, diarization and available voice embeddings; fresh candidate visual evidence; original visual evidence from checkpoint.'
            output.write_text(json.dumps(result,indent=2)+'\n')
            output.with_suffix('.txt').write_text('\n'.join(f"[{s['start']+offset:.2f}-{s['end']+offset:.2f}] {s['final_speaker']} ({s['final_confidence']:.2f}): {s['text']}" for s in result['segments'])+'\n')
            print('COMPLETED',clip,reference,flush=True)

if __name__=='__main__':main()
