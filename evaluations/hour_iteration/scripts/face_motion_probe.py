"""Track face identity and mouth geometry; activity hints do not prove speech."""
import json,time
from pathlib import Path
import cv2,numpy as np,onnxruntime as ort
from insightface.app import FaceAnalysis
p=Path('outputs/hour_iteration/face_probe');p.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
analyzer=FaceAnalysis(name='buffalo_l',providers=['CPUExecutionProvider'],sess_options=options,allowed_modules=['detection','recognition','landmark_3d_68']);analyzer.prepare(ctx_id=-1,det_size=(640,640))
ref=np.load('/home/think/projects/whisperx_diarization/references/youtube-v1/face_embeddings.npy');ref=ref/np.linalg.norm(ref,axis=1,keepdims=True);target=ref.mean(axis=0);target/=np.linalg.norm(target)
cap=cv2.VideoCapture('work/hour_iteration/face_probe.mp4');rows=[];tracks={};index=0;start=time.monotonic()
while True:
 ok,frame=cap.read()
 if not ok:break
 seconds=1086+index/4
 for face in analyzer.get(frame):
  if face.embedding is None:continue
  v=face.embedding/np.linalg.norm(face.embedding);similarity=float(v@target);landmarks=getattr(face,'landmark_3d_68',None);aperture=None
  if landmarks is not None:
   width=np.linalg.norm(landmarks[60]-landmarks[64])
   if width>1e-6:aperture=float(np.linalg.norm(landmarks[62]-landmarks[66])/width)
  # Identity clusters, not speaker labels. Current scene has a printed target face.
  matches=[(float(v@r['centroid']),name) for name,r in tracks.items()]
  match=max(matches,default=(-1,None))
  name=match[1] if match[0]>=.5 else f'FACE_{len(tracks):02d}'
  if name not in tracks:tracks[name]={'vectors':[],'centroid':v}
  tracks[name]['vectors'].append(v);centroid=np.mean(tracks[name]['vectors'],axis=0);tracks[name]['centroid']=centroid/np.linalg.norm(centroid)
  rows.append({'time':seconds,'face_track':name,'target_face_similarity':similarity,'bbox':face.bbox.tolist(),'det_score':float(face.det_score),'mouth_aperture':aperture})
 index+=1
 if index%40==0:
  (p/'face_observations.json').write_text(json.dumps(rows,indent=2)+'\n');print('Face frames',index,'elapsed',round(time.monotonic()-start),flush=True)
cap.release();(p/'face_observations.json').write_text(json.dumps(rows,indent=2)+'\n')
summary={name:{'observations':len(r['vectors'])} for name,r in tracks.items()};(p/'summary.json').write_text(json.dumps({'face_tracks':summary,'sample_rate_hz':4,'source_offset':1086,'active_speaker_verified':False,'model_providers':{name:model.session.get_providers() for name,model in analyzer.models.items()}},indent=2)+'\n');print('Face probe complete',summary,flush=True)
