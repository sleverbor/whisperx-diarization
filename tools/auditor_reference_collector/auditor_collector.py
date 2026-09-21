"""Local interactive auditor enrollment. Bind to loopback; keep library private."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import threading
from urllib.parse import urlparse,parse_qs,unquote
import uuid
from auditor_enrollment import rank_candidates,decide,export_reference,compare_voice,voice_reference,approved_short
from auditor_collector_engine import Engine,decode


class Library:
    def __init__(self,root,device,model):
        self.root=root;root.mkdir(parents=True,exist_ok=True)
        self.path=root/'session.json';self.lock=threading.RLock()
        self.data=json.loads(self.path.read_text()) if self.path.exists() else {'version':1,'seeds':[],'candidates':[]}
        self.engine=Engine(root,device,model);self.job={'running':False,'message':'Ready','error':None}
        self.comparison=None
    def save(self):
        temp=self.path.with_suffix('.tmp');temp.write_text(json.dumps(self.data,indent=2)+'\n');temp.replace(self.path)
    def progress(self,message):
        with self.lock:self.job['message']=message
    def background(self,fn):
        with self.lock:
            if self.job['running']:raise ValueError('Wait for the current job to finish')
            self.job={'running':True,'message':'Starting','error':None};self.comparison=None
        def work():
            try:fn()
            except Exception as error:
                with self.lock:self.job['error']=str(error);self.job['message']='Job failed; earlier decisions are saved'
            finally:
                with self.lock:self.job['running']=False
        threading.Thread(target=work,daemon=True).start()
    def checkpoint(self,row):
        with self.lock:
            # Rescanning a window does not duplicate already reviewed intervals.
            exists=any(x['source_url']==row['source_url'] and abs(x['source_start']-row['source_start'])<.05 and abs(x['source_end']-row['source_end'])<.05 for x in self.data['candidates'])
            if not exists:self.data['candidates'].append(row);self.save()
    def state(self,goal):
        with self.lock:
            candidates=rank_candidates(self.data['candidates'],self.data['candidates'],goal)
            filtered=[{k:v for k,v in r.items() if k not in ('voice_vector','face_vector')} for r in candidates]
            return {'job':dict(self.job),'candidates':filtered,'seeds':len(self.data['seeds']),
                'long':len(voice_reference(self.data['candidates'])),'short':len(approved_short(self.data['candidates'])),
                'reviewed':sum(bool(r.get('voice_decision')) for r in self.data['candidates']),
                'comparison':self.comparison,'last_channel_url':self.data.get('last_channel_url','')}
    def upload(self,body,directory,allowed):
        extension=Path(body['name']).suffix.lower()
        if extension not in allowed:raise ValueError('Unsupported file type')
        data=base64.b64decode(body['data'],validate=True)
        if len(data)>20*1024*1024:raise ValueError('File exceeds 20 MB')
        folder=self.root/directory;folder.mkdir(exist_ok=True)
        path=folder/(uuid.uuid4().hex+extension);path.write_bytes(data);return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library-dir',type=Path,default=Path('auditor-library'))
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--asr-model',default='small')
    parser.add_argument('--open-browser',action='store_true')
    args=parser.parse_args()
    library=Library(args.library_dir.expanduser().resolve(),args.device,args.asr_model)
    token=secrets.token_urlsafe(32)
    html=(Path(__file__).parent/'collector/index.html').read_text().replace('__TOKEN__',json.dumps(token))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass # Do not log the local access token.
        def authorized(self):
            return secrets.compare_digest(self.headers.get('X-Collector-Token','') or parse_qs(urlparse(self.path).query).get('token',[''])[0],token)
        def reply(self,value,status=200):
            raw=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if not self.authorized():self.reply({'error':'Open the local link printed in the terminal'},403);return
            parsed=urlparse(self.path)
            if parsed.path=='/':
                raw=html.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(raw)));self.send_header('Referrer-Policy','no-referrer');self.end_headers();self.wfile.write(raw)
            elif parsed.path=='/api/state':self.reply(library.state(parse_qs(parsed.query).get('goal',['balanced'])[0]))
            elif parsed.path.startswith('/media/'):
                path=(library.root/unquote(parsed.path[len('/media/'):])).resolve()
                if not path.is_relative_to(library.root) or not path.is_file() or path.suffix.lower() not in ('.mp4','.wav','.jpg','.png','.webp','.jpeg'):
                    self.reply({'error':'Media not found'},404);return
                size=path.stat().st_size;start=0;end=size-1;partial=False
                requested=self.headers.get('Range','')
                if requested.startswith('bytes='):
                    import re
                    m=re.fullmatch(r'bytes=(\d+)-(\d*)',requested)
                    if not m:self.reply({'error':'Unsupported range'},416);return
                    start=int(m[1]);end=min(size-1,int(m[2])) if m[2] else size-1;partial=True
                    if start>end:self.reply({'error':'Invalid range'},416);return
                self.send_response(206 if partial else 200);self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
                if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
                self.end_headers()
                try:
                    with path.open('rb') as stream:
                        stream.seek(start);remaining=end-start+1
                        while remaining:
                            chunk=stream.read(min(65536,remaining))
                            if not chunk:break
                            self.wfile.write(chunk);remaining-=len(chunk)
                except (BrokenPipeError,ConnectionResetError):pass
            else:self.reply({'error':'Not found'},404)
        def do_POST(self):
            if not self.authorized():self.reply({'error':'Unauthorized'},403);return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if size>28*1024*1024:raise ValueError('Request too large')
                body=json.loads(self.rfile.read(size));route=urlparse(self.path).path
                if route=='/api/seed':
                    if not body.get('confirmed'):raise ValueError('Confirm the images show the auditor')
                    with library.lock:
                        if library.job['running']:raise ValueError('Wait for current job')
                    path=library.upload(body,'seeds',('.png','.jpg','.jpeg','.webp'))
                    def add_seed():
                        library.progress('Analyzing confirmed seed image; initial model downloads may take time')
                        row=library.engine.seed(path)
                        with library.lock:
                            if not any(s['sha256']==row['sha256'] for s in library.data['seeds']):library.data['seeds'].append(row);library.save()
                        library.progress('Confirmed seed image saved')
                    library.background(add_seed)
                elif route=='/api/seed-url':
                    if not body.get('confirmed'):raise ValueError('Confirm this image shows the auditor')
                    url=body['url'].strip()
                    def add_url_seed():
                        from collector_image_download import download_image
                        library.progress('Downloading your selected image into the private library')
                        data=download_image(url)
                        folder=library.root/'seeds';folder.mkdir(exist_ok=True)
                        path=folder/(uuid.uuid4().hex+'.jpg');path.write_bytes(data)
                        row=library.engine.seed(path);row['source_url']=url
                        with library.lock:
                            if not any(s['sha256']==row['sha256'] for s in library.data['seeds']):library.data['seeds'].append(row);library.save()
                        library.progress('Selected image saved and confirmed')
                    library.background(add_url_seed)
                elif route=='/api/voice-seed':
                    if not body.get('confirmed'):raise ValueError('Confirm all speech is the auditor')
                    with library.lock:
                        if library.job['running']:raise ValueError('Wait for current job')
                    path=library.upload(body,'voice-seeds',('.wav','.mp3','.m4a','.mp4','.flac','.ogg'))
                    def add_voice():
                        library.progress('Analyzing operator-confirmed voice seed')
                        audio=decode(path);duration=len(audio)/16000
                        if not .3<=duration<=8:raise ValueError('Voice seed must be 0.3–8 seconds; use 2+ seconds for the main reference')
                        ident=uuid.uuid4().hex
                        row={'id':ident,'source_url':'operator-supplied:'+path.name,'source_start':0,'source_end':duration,
                             'duration':duration,'text':body.get('text',''),'confirmed_text':body.get('text',''),
                             'voice_vector':library.engine.voice_vector(audio),'face_vector':None,
                             'face_similarity':None,'view_hint':None,'clip':str(path.relative_to(library.root)),
                             'audio':str(path.relative_to(library.root)),'thumbnail':None,'operator_supplied_identity':True}
                        row=decide(row,'yes','not_visible',body.get('style','normal'),body.get('text',''))
                        library.checkpoint(row);library.progress('Confirmed voice seed saved')
                    library.background(add_voice)
                elif route=='/api/scan':
                    from auditor_enrollment import youtube_url
                    from auditor_collector_engine import video_window
                    video_window(body['url'],body.get('start',0),body.get('seconds',30))
                    url=youtube_url(body['url']);limit=int(body.get('video_limit',3));start=float(body.get('start',0));seconds=float(body.get('seconds',30))
                    with library.lock:
                        library.data['last_channel_url']=url;library.save()
                    library.background(lambda:library.engine.scan(url,list(library.data['seeds']),limit,start,seconds,library.progress,library.checkpoint))
                elif route=='/api/decision':
                    with library.lock:
                        row=next(r for r in library.data['candidates'] if r['id']==body['id'])
                        new=decide(row,body['voice'],body['face'],body.get('style','normal'),body.get('text'))
                        library.data['candidates'][library.data['candidates'].index(row)]=new;library.save()
                elif route=='/api/trim':
                    with library.lock:row=dict(next(r for r in library.data['candidates'] if r['id']==body['id']))
                    start=float(body['start']);end=float(body['end'])
                    if not 0<=start<end<=row['duration'] or end-start<.3:raise ValueError('Trim within the clip; minimum 0.3 seconds')
                    def trim():
                        library.progress('Creating separately reviewable trimmed sample')
                        new=library.engine.candidate(library.root/row['clip'],start,end,body.get('text',row['text']),library.data['seeds'],row['source_url'],row['source_start'])
                        new['parent_candidate_id']=row['id'];library.checkpoint(new);library.progress('Trimmed candidate created; confirm its identity separately')
                    library.background(trim)
                elif route=='/api/export':
                    def export():
                        import datetime
                        name=datetime.datetime.now().strftime('reference-%Y%m%d-%H%M%S')
                        with library.lock:export_reference(library.data,library.root/name)
                        library.progress('Reference exported inside your private library: '+name)
                    library.background(export)
                elif route=='/api/compare':
                    with library.lock:
                        if library.job['running']:raise ValueError('Wait for current job')
                    path=library.upload(body,'queries',('.wav','.mp3','.m4a','.mp4','.flac','.ogg'))
                    def compare():
                        audio=decode(path)
                        if len(audio)>16000*8:raise ValueError('Comparison audio must be at most 8 seconds')
                        vector=library.engine.voice_vector(audio)
                        with library.lock:library.comparison=compare_voice(vector,library.data['candidates'],body.get('phrase',''))
                        library.progress('Comparison complete; query is not added to enrollment')
                    library.background(compare)
                else:raise ValueError('Unknown action')
                self.reply({'ok':True})
            except Exception as error:self.reply({'error':str(error)},400)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Open http://127.0.0.1:{args.port}/?token={token}',flush=True)
    print('Private library:',library.root,flush=True)
    if args.open_browser:
        import webbrowser
        webbrowser.open(f'http://127.0.0.1:{args.port}/?token={token}')
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
