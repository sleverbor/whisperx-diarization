"""Build an explicit allowlist archive; never include environment files or caches."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

FILES = ("chainofrules.py", "cloud_runtime.py", "requirements-kaggle.txt",
         "test_cloud_runtime.py", "test_short_answers.py", "video.mp4", "longer_2min.mp4",
         "voice_embeddings.npy", "face_embeddings.npy", "KAGGLE.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a new output filename to preserve an existing bundle")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name in FILES:
            path = root/name
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(block)
            archive.write(path, "diarization-test/"+name)
            manifest[name] = digest.hexdigest()
        # The latest confirmed milestone transcript, without embedding values or secrets.
        evidence = json.loads((root/"brief_exchange_evidence.json").read_text())
        transcript = "\n".join(f"[{s['start']:.2f}-{s['end']:.2f}] {s['final_speaker']} (strength={s['final_confidence']:.2f}): {s['text']}" for s in evidence["segments"])
        archive.writestr("diarization-test/milestone_transcript.txt", transcript+"\n")
        archive.writestr("diarization-test/bundle_manifest.json", json.dumps(manifest, indent=2)+"\n")
    print(f"Prepared private dataset archive: {output} ({output.stat().st_size/1024**2:.1f} MiB)")

if __name__ == "__main__":
    main()
