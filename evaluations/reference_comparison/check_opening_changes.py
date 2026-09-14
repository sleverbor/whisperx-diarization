import os,sys,json
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
sys.path.insert(0,'/home/think/projects/whisperx_diarization')
import chainofrules as c
c.torch.set_num_threads(2);c.cv2.setNumThreads(2)
options=c.ort.SessionOptions();options.intra_op_num_threads=2
model=c.FaceAnalysis(name='buffalo_l',providers=['CPUExecutionProvider'],sess_options=options)
model.prepare(ctx_id=-1,det_size=(640,640))
root=Path('/home/think/projects/whisperx_diarization');out=Path('outputs/reference_comparison')
old=json.loads((out/'opening_original.json').read_text());new=json.loads((out/'opening_candidate.json').read_text())
a=c.np.load(root/'face_embeddings.npy');a/=c.np.linalg.norm(a,axis=1,keepdims=True);centroid=c.normalize_vector(a.mean(0))
cap=c.cv2.VideoCapture(str(root/'short.mp4'));fps=cap.get(c.cv2.CAP_PROP_FPS)
checks=[]
for original,candidate in zip(old['segments'],new['segments']):
 if original['final_speaker']==candidate['final_speaker']:continue
 segment=c.TimelineSegment(original['start'],original['end'],original['text'],c.Baseline(**original['baseline']))
 c.collect_visual_evidence(segment,cap,fps,model,centroid)
 checks.append({'start':segment.start,'text':segment.text,'fresh_original_visual':[c.asdict(e) for e in segment.evidence]})
 print('CHECKED',segment.start,flush=True)
cap.release();(out/'opening_visual_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
