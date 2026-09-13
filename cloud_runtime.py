"""Small runtime helpers; attribution rules do not live here."""
import hashlib
import json
import os
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class StageCache:
    """Atomic, JSON-only checkpoints, isolated by inputs/code/runtime fingerprint."""
    def __init__(self, directory, fingerprint):
        self.root = None
        if directory:
            key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
            self.root = Path(directory) / key
            self.root.mkdir(parents=True, exist_ok=True)
            self.write("manifest", fingerprint)

    def read(self, name):
        if self.root is None:
            return None
        path = self.root / (name + ".json")
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def write(self, name, value):
        if self.root is None:
            return
        path = self.root / (name + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False))
        os.replace(temporary, path)

    def get(self, name, compute):
        value = self.read(name)
        if value is not None:
            print(f"Reusing checkpoint: {name}")
            return value
        value = compute()
        self.write(name, value)
        return value


def create_face_analyzer(device, ort, factory):
    if device == "cuda" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    use_cuda = device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    analyzer = factory(name="buffalo_l", providers=providers)
    analyzer.prepare(ctx_id=0 if use_cuda else -1, det_size=(640, 640))
    actual = {name: model.session.get_providers() for name, model in analyzer.models.items()
              if getattr(model, "session", None) is not None}
    print(f"Face analysis actual providers: {actual}")
    if device == "cuda" and (not actual or any("CUDAExecutionProvider" not in value for value in actual.values())):
        print("WARNING: one or more face models are using CPU; check onnxruntime-gpu/CUDA libraries.")
    return analyzer, actual
