#!/usr/bin/env python3
"""Run contextual target-speaker extraction on a fixed labeled benchmark."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import types

import numpy as np

from review_overlap_extraction import rms, transcribe, unit


def crop_context_output(wave, sample_rate, window_start, interval_start, interval_end):
    left = round((interval_start - window_start) * sample_rate)
    right = round((interval_end - window_start) * sample_rate)
    return np.asarray(wave, dtype=np.float32)[max(0, left):max(0, right)]


def token_f1(reference, hypothesis):
    words=lambda x: re.findall(r"[a-z0-9']+",str(x).casefold())
    ref, hyp=words(reference),words(hypothesis)
    if not ref or not hyp:return 0.0
    remaining=list(ref);hits=0
    for word in hyp:
        if word in remaining:hits+=1;remaining.remove(word)
    precision=hits/len(hyp);recall=hits/len(ref)
    return 2*precision*recall/max(precision+recall,1e-9)


def main():
    p=argparse.ArgumentParser();p.add_argument("--video",type=Path,required=True);p.add_argument("--labels",type=Path,required=True);p.add_argument("--enrollment",type=Path,required=True);p.add_argument("--voice-priors",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--context",type=float,action="append",default=[]);p.add_argument("--device",default="cpu");p.add_argument("--whisper-model",default="large-v2");p.add_argument("--wesep-model-dir",type=Path)
    a=p.parse_args();contexts=a.context or [3.0,5.0];a.output_dir.mkdir(parents=True,exist_ok=True)
    import soundfile as sf
    import torch
    # WeSep imports an unused UMAP diarization dependency. Avoid initializing its
    # Numba cache in portable/local environments; extraction does not use UMAP.
    if "umap" not in sys.modules:
        module=types.ModuleType("umap");module.UMAP=object;sys.modules["umap"]=module
    import wesep
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import SpeakerRecognition
    labels=json.loads(a.labels.read_text())["labels"]
    source=a.output_dir/"source-16khz.wav"
    subprocess.run(["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-y","-i",str(a.video),"-vn","-ac","1","-ar","16000",str(source)],check=True)
    full,sr=sf.read(source,dtype="float32");assert sr==16000 and full.ndim==1
    extractor=(wesep.load_model_local(str(a.wesep_model_dir)) if a.wesep_model_dir else wesep.load_model("english"));extractor.set_device(a.device);extractor.set_vad(True);extractor.set_output_norm(False)
    priors=np.load(a.voice_priors);target=unit(np.mean(np.stack([unit(x) for x in priors]),axis=0))
    spkdir=Path("pretrained_models/spkrec-ecapa-voxceleb")
    speaker=SpeakerRecognition.from_hparams(source=str(spkdir),savedir=str(spkdir),run_opts={"device":a.device})
    whisper=WhisperModel(a.whisper_model,device="cuda" if a.device.startswith("cuda") else "cpu",compute_type="float16" if a.device.startswith("cuda") else "int8")
    audio=a.output_dir/"audio";audio.mkdir(exist_ok=True);rows=[]
    def similarity(w):
        tensor=torch.from_numpy(np.asarray(w,dtype=np.float32)).unsqueeze(0)
        emb=unit(speaker.encode_batch(tensor).flatten().detach().cpu().numpy())
        return float(np.dot(target,emb))
    for label in labels:
        start,end=float(label["start"]),float(label["end"])
        for context in contexts:
            ws=max(0.0,start-context);we=min(len(full)/sr,end+context)
            window=full[round(ws*sr):round(we*sr)];stem=f"{label['exchange_id']}-context-{context:g}s"
            window_path=audio/f"{stem}-input.wav";sf.write(window_path,window,sr)
            tensor=extractor.extract_speech(str(window_path),str(a.enrollment))
            if tensor is None:
                rows.append({"exchange_id":label["exchange_id"],"baseline_index":label["baseline_index"],"context_seconds":context,"status":"no_speech"});continue
            separated=tensor[0].detach().cpu().numpy()
            cropped=crop_context_output(separated,sr,ws,start,end)
            expected=round((end-start)*sr)
            if len(cropped)<expected:cropped=np.pad(cropped,(0,expected-len(cropped)))
            cropped=cropped[:expected]
            path=audio/f"{stem}-target.wav";sf.write(path,cropped,sr)
            asr=transcribe(whisper,cropped);text=asr["text"]
            row={"exchange_id":label["exchange_id"],"baseline_index":label["baseline_index"],"start":start,"end":end,"context_seconds":context,"status":"completed","output_length_samples":len(separated),"input_length_samples":len(window),"alignment_preserved":len(separated)==len(window),"audio":str(path.relative_to(a.output_dir)),"target_similarity":similarity(cropped),"energy_retention":rms(cropped)/max(rms(full[round(start*sr):round(end*sr)]),1e-9),"transcription":asr,"target_word_f1":token_f1(label.get("target_words",""),text),"other_word_f1":token_f1(label.get("other_words",""),text)}
            rows.append(row);print(f"{stem}: sim={row['target_similarity']:.3f} target_f1={row['target_word_f1']:.2f} other_f1={row['other_word_f1']:.2f}: {text}",flush=True)
    report={"schema_version":1,"method":"contextual_wesep","labels":str(a.labels.resolve()),"labels_used_for_extraction":False,"contexts":contexts,"results":rows}
    (a.output_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":main()
