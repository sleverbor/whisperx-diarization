"""Higher-rate face tracking probe; identity/speech never inferred from presence."""
import json,time
from pathlib import Path
import cv2,numpy as np,onnxruntime as ort
from insightface.app import FaceAnalysis
p=Path('outputs/hour_iteration/dense_face');p.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
analyzer=FaceAnalysis(name='buffalo_l',providers=['CPUExecutionProvider'],sess_options=options,allowed_modules=['detection','recognition','landmark_3d_68']);analyzer.prepare(ctx_id=-1,det_size=(640,640))
cap=cv2.VideoCapture('/home/think/projects/whisperx_diarization/video.mp4');fps=cap.get(cv2.CAP_PROP_FPS);cap.set(cv2.CAP_PROP_POS_MSEC,1086*1000);rows=[];tracks={};count=0;started=time.monotonic()
def iou(a,b):
 x=max(a[0],b[0]);y=max(a[1],b[1]);r=min(a[2],b[2]);s=min(a[3],b[3]);inter=max(0,r-x)*max(0,s-y);return inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
while count<round(14*fps):
 ok,frame=cap.read()
 if not ok:break
 t=1086+count/fps
 if count%max(1,round(fps/15))==0:
  used=set()
  for face in analyzer.get(frame):
   if face.embedding is None:continue
   v=face.embedding/np.linalg.norm(face.embedding);box=face.bbox.tolist();candidates=[]
   for name,r in tracks.items():
    if t-r['last_time']>.3 or name in used:continue
    spatial=iou(box,r['bbox']);appearance=float(v@r['embedding'])
    if spatial>=.25 and appearance>=.1:candidates.append((spatial,name))
   match=max(candidates,default=(0,None));name=match[1] if match[1] else f'TRACK_{len(tracks):02d}';used.add(name);tracks[name]={'last_time':t,'bbox':box,'embedding':v}
   landmarks=getattr(face,'landmark_3d_68',None);width=np.linalg.norm(landmarks[60]-landmarks[64]) if landmarks is not None else 0;aperture=float(np.linalg.norm(landmarks[62]-landmarks[66])/width) if width>1e-6 else None
   pose=getattr(face,'pose',None);rows.append({'time':t,'face_track':name,'bbox':box,'det_score':float(face.det_score),'mouth_aperture':aperture,'pose':pose.tolist() if pose is not None else None})
  if count%round(fps*2)==0:(p/'observations.json').write_text(json.dumps(rows,indent=2)+'\n');print('Dense face',round(t,2),len(rows),'elapsed',round(time.monotonic()-started),flush=True)
 count+=1
cap.release();(p/'observations.json').write_text(json.dumps(rows,indent=2)+'\n');(p/'provenance.json').write_text(json.dumps({'source_start':1086,'source_duration':14,'source_fps':fps,'sample_every_frames':max(1,round(fps/15)),'tracking':'consecutive spatial overlap plus appearance, diagnostic only','speaker_identity_verified':False,'elapsed_seconds':time.monotonic()-started},indent=2)+'\n');print('Dense face complete',len(rows),flush=True)
