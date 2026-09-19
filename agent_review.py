#!/usr/bin/env python3
"""Local player and durable command surface for conversational diarization review."""

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import webbrowser


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def session_file(session, name):
    return Path(session) / name


def rows_from_source(source):
    data = load(source)
    rows = data.get("labels") or data.get("segments")
    if rows is None:
        raise ValueError("Expected a labels or segments array")
    result = []
    for number, row in enumerate(rows, 1):
        if "tier" in row and row.get("tier") != "separation_review":
            continue
        start, end = float(row["start"]), float(row["end"])
        result.append({
            "review_id": str(row.get("review_id", row.get("baseline_index", number))),
            "start": start, "end": end,
            "context_start": max(0.0, start - 4.0), "context_end": end + 4.0,
            "text": row.get("text", row.get("baseline_text", "")),
            "speaker": row.get("baseline_speaker", row.get("final_speaker")),
            "source": str(source),
        })
    return result


def init_session(session, source, video_id=None):
    session = Path(session); session.mkdir(parents=True, exist_ok=True)
    queue = rows_from_source(source)
    save(session / "session.json", {
        "schema_version": 1, "video_id": video_id, "created_at": now(),
        "source": str(Path(source).resolve()), "current_index": 0,
    })
    save(session / "queue.json", queue)
    save(session / "observations.json", [])
    save(session / "annotations.json", [])
    save(session / "control.json", {"revision": 0, "action": "load", "item": queue[0]})
    save(session / "player-state.json", {"media_time": None, "playing": False, "updated_at": None})
    return len(queue)


def current(session):
    meta = load(session_file(session, "session.json")); queue = load(session_file(session, "queue.json"), [])
    index = meta["current_index"]
    return meta, queue, queue[index] if queue else None


def control(session, action, **values):
    path = session_file(session, "control.json")
    prior = load(path, {"revision": 0})
    payload = {"revision": prior["revision"] + 1, "action": action, **values}
    save(path, payload); return payload


def select_item(session, index):
    meta, queue, _ = current(session)
    if not 0 <= index < len(queue): raise IndexError(index)
    meta["current_index"] = index; save(session_file(session, "session.json"), meta)
    return control(session, "load", item=queue[index])


def add_comment(session, text):
    _, _, item = current(session); state = load(session_file(session, "player-state.json"), {})
    rows = load(session_file(session, "observations.json"), [])
    row = {"observation_id": len(rows) + 1, "review_id": item["review_id"],
           "comment": text, "approximate_media_time": state.get("media_time"),
           "created_at": now(), "anchors": [], "status": "needs_grounding"}
    rows.append(row); save(session_file(session, "observations.json"), rows); return row


def add_anchor(session, observation_id, kind):
    state = load(session_file(session, "player-state.json"), {})
    if state.get("media_time") is None: raise ValueError("Player has not reported a media position")
    rows = load(session_file(session, "observations.json"), [])
    row = next((x for x in rows if x["observation_id"] == observation_id), None)
    if row is None: raise ValueError(f"Unknown observation {observation_id}")
    row["anchors"].append({"kind": kind, "media_time": state["media_time"], "created_at": now()})
    row["status"] = "grounded" if kind == "here" or {x["kind"] for x in row["anchors"]} >= {"start", "end"} else "needs_grounding"
    save(session_file(session, "observations.json"), rows); return row


def add_annotation(session, args):
    _, _, item = current(session); rows = load(session_file(session, "annotations.json"), [])
    row = {"annotation_id": len(rows) + 1, "review_id": item["review_id"],
           "observation_id": args.observation_id, "speaker": args.speaker,
           "text": args.text, "true_overlap": args.overlap,
           "attribution_basis": [x for x in args.basis.split(",") if x],
           "speaker_confidence": args.speaker_confidence,
           "word_confidence": args.word_confidence, "confirmed": True,
           "created_at": now()}
    rows.append(row); save(session_file(session, "annotations.json"), rows); return row


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>Agent Review Player</title>
<style>body{font:16px system-ui;max-width:980px;margin:24px auto;background:#101318;color:#eef2f6}video,audio{width:100%;max-height:62vh;background:#000}.bar{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}button,label{background:#283241;color:white;border:0;border-radius:8px;padding:9px 12px}.card{background:#1a202a;padding:16px;border-radius:12px;margin:12px 0}.muted{color:#aeb8c5}input[type=file]{max-width:300px}</style></head><body>
<h1>Conversational diarization review</h1><div class="card"><label>Choose the matching local video or audio <input id="file" type="file" accept="video/*,audio/*"></label></div>
<div id="media"></div><div class="bar"><button onclick="back()">−2 sec</button><button onclick="toggle()">Play/Pause</button><button onclick="loopClip()">Loop clip</button><button onclick="speed(.5)">0.5×</button><button onclick="speed(.75)">0.75×</button><button onclick="speed(1)">1×</button></div>
<div class="card"><b id="where">Waiting for queue…</b><p id="text"></p><span class="muted" id="state"></span></div>
<script>let m=null,item=null,lastRevision=-1,loop=false;
file.onchange=()=>{if(m)m.remove();let f=file.files[0];m=document.createElement(f.type.startsWith('audio')?'audio':'video');m.controls=true;m.src=URL.createObjectURL(f);media.replaceChildren(m)};
function back(){if(m)m.currentTime=Math.max(0,m.currentTime-2)}function toggle(){if(m)(m.paused?m.play():m.pause())}function speed(x){if(m)m.playbackRate=x}function loopClip(){loop=!loop;state.textContent=loop?'Loop enabled':''}
setInterval(()=>{if(m&&item&&loop&&m.currentTime>=item.context_end){m.currentTime=item.context_start;m.play()}},100);
setInterval(async()=>{if(m)await fetch('/api/player-state',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({media_time:m.currentTime,playing:!m.paused,rate:m.playbackRate})})},200);
setInterval(async()=>{let c=await(await fetch('/api/control')).json();if(c.revision===lastRevision)return;lastRevision=c.revision;if(c.item){item=c.item;where.textContent=`${item.start.toFixed(2)}–${item.end.toFixed(2)} sec · review ${item.review_id}`;text.textContent=item.text;if(m){m.currentTime=item.context_start;m.playbackRate=c.rate||1;if(c.autoplay)m.play()}}if(c.action==='play'&&m){if(c.time!=null)m.currentTime=c.time;m.play()}if(c.action==='pause'&&m)m.pause()},500);
</script></body></html>'''


def serve(session, host, port, open_browser):
    session = Path(session).resolve(); lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200, content_type="application/json"):
            body = value.encode() if isinstance(value, str) else json.dumps(value).encode()
            self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if self.path == "/": return self.reply(HTML, content_type="text/html; charset=utf-8")
            if self.path == "/api/control": return self.reply(load(session / "control.json", {}))
            if self.path == "/api/status":
                meta, queue, item = current(session)
                return self.reply({"session": meta, "item": item, "queue_size": len(queue), "player": load(session / "player-state.json", {})})
            return self.reply({"error": "not found"}, 404)
        def do_POST(self):
            if self.path != "/api/player-state": return self.reply({"error": "not found"}, 404)
            length = int(self.headers.get("Content-Length", 0)); data = json.loads(self.rfile.read(length) or b"{}")
            with lock: save(session / "player-state.json", {**data, "updated_at": now()})
            return self.reply({"ok": True})
        def log_message(self, *_): pass
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(url, flush=True)
    if open_browser: webbrowser.open(url)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    p=sub.add_parser("init"); p.add_argument("--session",required=True);p.add_argument("--source",required=True);p.add_argument("--video-id")
    p=sub.add_parser("serve"); p.add_argument("--session",required=True);p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8765);p.add_argument("--open",action="store_true")
    p=sub.add_parser("status");p.add_argument("--session",required=True)
    p=sub.add_parser("select");p.add_argument("--session",required=True);p.add_argument("index",type=int)
    p=sub.add_parser("next");p.add_argument("--session",required=True)
    p=sub.add_parser("play");p.add_argument("--session",required=True);p.add_argument("--time",type=float);p.add_argument("--rate",type=float,default=1.0)
    p=sub.add_parser("comment");p.add_argument("--session",required=True);p.add_argument("text")
    p=sub.add_parser("anchor");p.add_argument("--session",required=True);p.add_argument("observation_id",type=int);p.add_argument("kind",choices=["here","start","end"])
    p=sub.add_parser("annotate");p.add_argument("--session",required=True);p.add_argument("--observation-id",type=int);p.add_argument("--speaker",required=True);p.add_argument("--text",default="");p.add_argument("--overlap",choices=["yes","no","unclear"],required=True);p.add_argument("--basis",default="");p.add_argument("--speaker-confidence",type=float);p.add_argument("--word-confidence",type=float)
    p=sub.add_parser("export");p.add_argument("--session",required=True);p.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="init": print(json.dumps({"queue_size":init_session(args.session,args.source,args.video_id)}))
    elif args.command=="serve": serve(args.session,args.host,args.port,args.open)
    elif args.command=="status":
        meta,queue,item=current(args.session);print(json.dumps({"session":meta,"item":item,"queue_size":len(queue),"player":load(session_file(args.session,"player-state.json"),{})},indent=2))
    elif args.command=="select": print(json.dumps(select_item(args.session,args.index),indent=2))
    elif args.command=="next":
        meta,queue,_=current(args.session);print(json.dumps(select_item(args.session,min(meta["current_index"]+1,len(queue)-1)),indent=2))
    elif args.command=="play": print(json.dumps(control(args.session,"play",time=args.time,rate=args.rate,autoplay=True),indent=2))
    elif args.command=="comment": print(json.dumps(add_comment(args.session,args.text),indent=2))
    elif args.command=="anchor": print(json.dumps(add_anchor(args.session,args.observation_id,args.kind),indent=2))
    elif args.command=="annotate": print(json.dumps(add_annotation(args.session,args),indent=2))
    elif args.command=="export":
        meta,queue,_=current(args.session); payload={"schema_version":1,"session":meta,"queue":queue,"observations":load(session_file(args.session,"observations.json"),[]),"annotations":load(session_file(args.session,"annotations.json"),[])};save(args.output,payload);print(args.output)


if __name__ == "__main__": main()
