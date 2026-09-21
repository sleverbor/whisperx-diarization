"""Bounded channel sampling and local media analysis for manual auditor enrollment."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid
import numpy as np
from auditor_enrollment import youtube_url, unit, cosine_scores


def run(args, timeout=900):
    try:
        return subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.CalledProcessError as error:
        detail=(error.stderr or b'').decode(errors='replace')[-1800:]
        raise RuntimeError('Media command failed: '+detail) from error


def decode(path):
    data = run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-vn','-ar','16000','-ac','1','-f','f32le','pipe:1']).stdout
    audio = np.frombuffer(data,dtype='<f4').copy()
    if not len(audio): raise ValueError('Media contains no audio')
    return audio


def speech_chunks(segments, maximum=8):
    chunks=[]
    for s in segments:
        if s.end-s.start <= maximum:
            chunks.append({'start':s.start,'end':s.end,'text':s.text})
            continue
        current=[]
        for w in s.words or []:
            if current and w.end-current[0].start>maximum:
                chunks.append({'start':current[0].start,'end':current[-1].end,'text':''.join(x.word for x in current)})
                current=[]
            current.append(w)
        if current: chunks.append({'start':current[0].start,'end':current[-1].end,'text':''.join(x.word for x in current)})
    return [x for x in chunks if .3<=x['end']-x['start']<=maximum and x['text'].strip()]


class Engine:
    def __init__(self, root, device='auto', model='small'):
        self.root=Path(root);self.device=device;self.model_name=model
        self.voice=None;self.face=None;self.asr=None

    def models(self, asr=False):
        if self.voice is None:
            import torch
            import cv2
            import onnxruntime as ort
            from speechbrain.inference.speaker import SpeakerRecognition
            from insightface.app import FaceAnalysis
            from cloud_runtime import create_face_analyzer
            torch.set_num_threads(2);cv2.setNumThreads(2)
            self.device='cuda' if self.device=='auto' and torch.cuda.is_available() else ('cpu' if self.device=='auto' else self.device)
            if self.device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
            self.voice=SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',
                savedir=str(self.root/'models/ecapa'),run_opts={'device':'cuda:0' if self.device=='cuda' else 'cpu'})
            self.face,self.providers=create_face_analyzer(self.device,ort,
                lambda **kwargs: FaceAnalysis(allowed_modules=['detection','recognition'],**kwargs))
        if asr and self.asr is None:
            from faster_whisper import WhisperModel
            self.asr=WhisperModel(self.model_name,device=self.device,compute_type='float16' if self.device=='cuda' else 'int8')

    def voice_vector(self, audio):
        import torch
        self.models()
        if len(audio)<4800: raise ValueError('Audio must be at least 0.3 seconds')
        with torch.no_grad():
            v=self.voice.encode_batch(torch.from_numpy(audio.copy()).unsqueeze(0).to(self.device)).flatten().cpu().numpy()
        return unit(v).tolist()

    def image(self, image):
        self.models()
        faces=[f for f in self.face.get(image) if f.det_score>=.7 and min(f.bbox[2]-f.bbox[0],f.bbox[3]-f.bbox[1])>=60]
        if len(faces)!=1:return None,None
        face=faces[0];view=None
        if getattr(face,'kps',None) is not None:
            left,right,nose=face.kps[:3]
            span=float(np.linalg.norm(right-left))
            displacement=abs(float(nose[0]-(left[0]+right[0])/2))/max(span,1)
            view='side' if displacement>.3 else 'front' if displacement<.18 else 'angled'
        return unit(face.embedding).tolist(),view

    def seed(self, path):
        import cv2
        image=cv2.imread(str(path))
        if image is None:raise ValueError('Image could not be decoded')
        vector,_=self.image(image)
        if vector is None:raise ValueError('Seed image must contain one clear face at least 60 pixels wide/high')
        return {'id':uuid.uuid4().hex,'image':str(path.relative_to(self.root)),
                'face_vector':vector,'face_decision':'yes','operator_supplied_identity':True,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

    def candidate(self, source, start, end, text, seeds, url, offset):
        import cv2
        if start<0 or end<=start or end-start>8:raise ValueError('Candidate interval must be between 0.3 and 8 seconds')
        ident=uuid.uuid4().hex
        directory=self.root/'clips';directory.mkdir(exist_ok=True)
        clip=directory/f'{ident}.mp4';audio_path=directory/f'{ident}.wav';thumb=directory/f'{ident}.jpg'
        run(['ffmpeg','-nostdin','-v','error','-ss',str(start),'-i',str(source),'-t',str(end-start),
             '-c:v','libx264','-preset','ultrafast','-crf','24','-c:a','aac','-movflags','+faststart',str(clip)])
        audio=decode(clip)
        run(['ffmpeg','-nostdin','-v','error','-i',str(clip),'-vn','-ar','16000','-ac','1',str(audio_path)])
        cap=cv2.VideoCapture(str(clip));cap.set(cv2.CAP_PROP_POS_MSEC,(end-start)*500);ok,image=cap.read();cap.release()
        face_vector=None;view=None
        if ok:
            cv2.imwrite(str(thumb),image);face_vector,view=self.image(image)
        similarity=max(cosine_scores(face_vector,[x['face_vector'] for x in seeds])) if face_vector and seeds else None
        return {'id':ident,'source_url':url,'source_start':offset+start,'source_end':offset+end,
                'duration':len(audio)/16000,'text':text.strip(),'clip':str(clip.relative_to(self.root)),
                'audio':str(audio_path.relative_to(self.root)),'thumbnail':str(thumb.relative_to(self.root)) if ok else None,
                'voice_vector':self.voice_vector(audio),'face_vector':face_vector,
                'face_similarity':similarity,'view_hint':view,'voice_decision':None,'face_decision':None,
                'note':'One midpoint face sample and ASR text are suggestions, not speaker identity.'}

    def scan(self,url,seeds,video_limit,start,seconds,progress,checkpoint):
        return scan_video_window(self,url,seeds,start,seconds,progress,checkpoint)


def channel_videos_url(url):
    from urllib.parse import urlparse
    p=urlparse(youtube_url(url));parts=p.path.strip('/').split('/')
    if (len(parts)==1 and parts[0].startswith('@')) or (len(parts)==2 and parts[0] in ('channel','c','user')):
        return p._replace(path=p.path.rstrip('/')+'/videos').geturl()
    return url


def video_entries(metadata,limit):
    import re
    result=[];seen=set()
    def visit(item):
        if not isinstance(item,dict):return
        ident=item.get('id','')
        if re.fullmatch(r'[A-Za-z0-9_-]{11}',ident) and ident not in seen:
            seen.add(ident);result.append(item)
        for entry in item.get('entries') or []:visit(entry)
    visit(metadata)
    return result[:limit]


def matched_scenes(observations,duration,step=2,minimum=4):
    scenes=[];run_frames=[]
    def finish():
        if not run_frames:return
        start=max(0,run_frames[0]['time']-step/2);end=min(duration,run_frames[-1]['time']+step/2)
        if end-start>=minimum and len(run_frames)>=3:
            scenes.append({'start':start,'end':end,'matched_frame_count':len(run_frames),
                           'minimum_similarity':min(x['similarity'] for x in run_frames)})
    for item in observations:
        if item['matched']:
            if run_frames and item['time']-run_frames[-1]['time']>step*1.5:finish();run_frames=[]
            run_frames.append(item)
        else:finish();run_frames=[]
    finish();return scenes


def sampled_video_frames(path, step=2):
    """Decode AV1 and other source codecs through FFmpeg, rather than OpenCV."""
    import tempfile
    meta=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height:format=duration','-of','json',str(path)]).stdout)
    width,height=meta['streams'][0]['width'],meta['streams'][0]['height']
    duration=float(meta['format']['duration'])
    def frames():
        with tempfile.TemporaryFile() as errors:
            process=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-i',str(path),
                '-vf',f'fps=1/{step}:start_time=0','-an','-f','rawvideo','-pix_fmt','bgr24','pipe:1'],
                stdout=subprocess.PIPE,stderr=errors)
            count=0;size=width*height*3
            try:
                while True:
                    data=bytearray()
                    while len(data)<size:
                        part=process.stdout.read(size-len(data))
                        if not part:break
                        data.extend(part)
                    if not data:break
                    if len(data)!=size:raise ValueError('Incomplete decoded video frame')
                    yield count*step,np.frombuffer(data,dtype=np.uint8).reshape(height,width,3)
                    count+=1
                status=process.wait()
                if status or not count:
                    errors.seek(0)
                    raise RuntimeError('Video decoding failed: '+errors.read().decode(errors='replace')[-1800:])
            finally:
                process.stdout.close()
                if process.poll() is None:process.terminate()
                process.wait()
    return duration,frames()


def scan_whole_videos(self,url,seeds,video_limit,start,seconds,progress,checkpoint):
    # Keep the old signature for saved callers; start/seconds do not restrict
    # whole-video discovery. The UI now makes the mode explicit.
    url=channel_videos_url(url)
    if not seeds:raise ValueError('Add at least one confirmed seed image first')
    if not 1<=video_limit<=10:raise ValueError('Choose between 1 and 10 videos')
    progress('Listing individual channel videos')
    metadata=json.loads(run([sys.executable,'-m','yt_dlp','--flat-playlist','--playlist-end',str(video_limit),
        '--dump-single-json','--skip-download','--no-warnings',url],timeout=120).stdout)
    entries=video_entries(metadata,video_limit)
    if not entries:raise ValueError('Channel listing returned no individual video IDs. Try the channel /videos URL or a direct video URL.')
    sources=self.root/'sources';sources.mkdir(exist_ok=True)
    (sources/'last-channel-listing.json').write_text(json.dumps(metadata,indent=2))
    self.models()
    total_scenes=0;total_candidates=0
    for index,item in enumerate(entries):
        ident=item['id'];video_url='https://www.youtube.com/watch?v='+ident
        destination=sources/(ident+'_full480.mp4')
        progress(f'Downloading whole video {index+1}/{len(entries)} at up to 480p')
        if not destination.exists():
            run([sys.executable,'-m','yt_dlp','--no-playlist','--no-warnings',
                '-f','bv*[height<=480]+ba/b[height<=480]/b','--merge-output-format','mp4','--recode-video','mp4',
                '-o',str(destination),video_url],timeout=1800)
        duration,frame_stream=sampled_video_frames(destination)
        observations=[];step=2
        scene_path=destination.with_suffix('.face-scenes.json')
        fingerprint=hashlib.sha256(json.dumps([s['face_vector'] for s in seeds],sort_keys=True).encode()).hexdigest()
        cached=json.loads(scene_path.read_text()) if scene_path.exists() else None
        if cached and cached.get('scanner_version')==2 and cached.get('seed_fingerprint')==fingerprint:
            scenes=cached['scenes'];progress(f'Reusing face-scene scan for video {index+1}')
        else:
            for n,(t,image) in enumerate(frame_stream):
                vector,view=self.image(image)
                similarity=max(cosine_scores(vector,[s['face_vector'] for s in seeds])) if vector else -1
                observations.append({'time':float(t),'matched':bool(vector is not None and similarity>=.4),
                                     'similarity':similarity,'view_hint':view})
                if n%15==0:progress(f'Inspecting whole video {index+1}/{len(entries)}: {t/60:.1f}/{duration/60:.1f} minutes')
            scenes=matched_scenes(observations,duration,step)
            scene_path.write_text(json.dumps({'scanner_version':2,'seed_fingerprint':fingerprint,'frame_step_seconds':step,
                'match_cutoff':.4,'scenes':scenes,'observations':observations},indent=2))
        total_scenes+=len(scenes)
        progress(f'Video {index+1}: found {len(scenes)} sustained single-auditor-face scenes; extracting speech')
        for scene_index,scene in enumerate(scenes):
            self.models(asr=True)
            # Decode only matched scenes; do not transcribe unrelated whole-video audio.
            for left in np.arange(scene['start'],scene['end'],60):
                right=min(float(left)+60,scene['end'])
                audio_bytes=run(['ffmpeg','-nostdin','-v','error','-ss',str(left),'-i',str(destination),'-t',str(right-left),
                                 '-vn','-ar','16000','-ac','1','-f','f32le','pipe:1']).stdout
                audio=np.frombuffer(audio_bytes,dtype='<f4').copy()
                segments,_=self.asr.transcribe(audio,word_timestamps=True,vad_filter=True,condition_on_previous_text=False)
                for chunk in speech_chunks(list(segments)):
                    progress(f'Video {index+1}, face scene {scene_index+1}/{len(scenes)}: preparing speech candidate')
                    row=self.candidate(destination,float(left)+chunk['start'],float(left)+chunk['end'],chunk['text'],seeds,video_url,0)
                    if row.get('face_similarity') is None or row['face_similarity']<.4:continue
                    row['visual_discovery']={**scene,'frame_step_seconds':step,'note':'Only the auditor face detected in sampled frames; off-camera or overlapping voices still require manual review.'}
                    checkpoint(row);total_candidates+=1
        progress(f'Video {index+1}/{len(entries)} complete: {total_candidates} candidate clips prepared so far')
    progress(f'Whole-video scan complete: {len(entries)} videos, {total_scenes} matching face scenes, {total_candidates} candidate clips. Face-only scenes do not prove who speaks.')


def video_window(url,start,seconds):
    import math,re
    from urllib.parse import urlparse,parse_qs
    url=youtube_url(url);parsed=urlparse(url)
    ident=parsed.path.strip('/') if parsed.hostname in ('youtu.be','www.youtu.be') else parse_qs(parsed.query).get('v',[''])[0]
    if not ident and parsed.path.startswith(('/shorts/','/embed/')):ident=parsed.path.split('/')[2]
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}',ident):raise ValueError('Provide an individual YouTube video URL')
    start,seconds=float(start),float(seconds)
    if not math.isfinite(start) or not math.isfinite(seconds) or start<0 or seconds<.3 or seconds>600:
        raise ValueError('Start must be nonnegative; duration must be between 0.3 and 600 seconds')
    return 'https://www.youtube.com/watch?v='+ident,ident,start,seconds


def scan_video_window(self,url,seeds,start,seconds,progress,checkpoint):
    url,ident,start,seconds=video_window(url,start,seconds)
    sources=self.root/'sources';sources.mkdir(exist_ok=True)
    key=hashlib.sha256(f'{ident}:{start}:{seconds}'.encode()).hexdigest()[:16]
    destination=sources/f'{ident}_window_{key}.mp4'
    progress(f'Downloading selected window: {start:g}–{start+seconds:g} seconds')
    if not destination.exists():
        run([sys.executable,'-m','yt_dlp','--no-playlist','--no-warnings',
             '-f','bv*[height<=480]+ba/b[height<=480]/b','--download-sections',f'*{start}-{start+seconds}',
             '--force-keyframes-at-cuts','--merge-output-format','mp4','-o',str(destination),url],timeout=1800)
    self.models(asr=True)
    progress('Transcribing the selected window and preparing clips for review')
    audio=decode(destination)[:round(seconds*16000)]
    segments,_=self.asr.transcribe(audio,word_timestamps=True,vad_filter=True,condition_on_previous_text=False)
    count=0
    for chunk in speech_chunks(list(segments)):
        right=min(chunk['end'],len(audio)/16000)
        if right-chunk['start']<.3:continue
        row=self.candidate(destination,chunk['start'],right,chunk['text'],seeds,url,start)
        row['note']='User-selected video window. Voice and face identity require independent confirmation.'
        checkpoint(row);count+=1
        progress(f'Prepared {count} clips from your selected window')
    progress(f'Selected-window scan complete: {count} clips ready for review')
