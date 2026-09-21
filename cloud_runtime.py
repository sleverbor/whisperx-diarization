"""Small runtime helpers; attribution rules do not live here."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def atomic_json(path, value):
    """Write valid JSON as one atomic replacement, including process flush."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


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
        atomic_json(path, value)

    def get(self, name, compute):
        value = self.read(name)
        if value is not None:
            print(f"Reusing checkpoint: {name}")
            return value
        value = compute()
        self.write(name, value)
        return value


class ResumableWorkSet:
    """Per-item atomic checkpoints with configuration and artifact validation.

    Only a completed JSON envelope is reusable. Temporary writes, exceptions,
    absent artifacts, and changed artifact bytes all cause that item to rerun.
    Different input/model/configuration fingerprints use different directories.
    """
    schema_version = 1

    def __init__(self, directory, fingerprint):
        self.enabled = bool(directory)
        self.fingerprint = fingerprint
        self.key = hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        self.root = Path(directory) / self.key if directory else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            manifest = self.root / "manifest.json"
            if manifest.exists():
                saved = json.loads(manifest.read_text())
                if saved.get("fingerprint") != fingerprint:
                    raise ValueError("Checkpoint manifest fingerprint mismatch")
            else:
                atomic_json(manifest, {"schema_version": self.schema_version,
                                       "fingerprint": fingerprint})

    @staticmethod
    def _filename(item_id):
        readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(item_id)).strip("-")[:80] or "item"
        suffix = hashlib.sha256(str(item_id).encode()).hexdigest()[:12]
        return f"{readable}-{suffix}.json"

    def _path(self, item_id):
        return self.root / "items" / self._filename(item_id)

    def read(self, item_id):
        if self.root is None:
            return None
        path = self._path(item_id)
        if not path.is_file():
            return None
        envelope = json.loads(path.read_text())
        if (envelope.get("schema_version") != self.schema_version
                or envelope.get("item_id") != str(item_id)
                or envelope.get("status") != "complete"):
            return None
        for artifact in envelope.get("artifacts", []):
            artifact_path = Path(artifact["path"])
            valid = artifact_path.is_file() and file_digest(artifact_path) == artifact["sha256"]
            cached = self.root / artifact.get("cached_path", "")
            if not valid and cached.is_file() and file_digest(cached) == artifact["sha256"]:
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
                shutil.copy2(cached, temporary)
                os.replace(temporary, artifact_path)
                valid = True
            if not valid:
                return None
        return envelope.get("result")

    def complete(self, item_id, result, artifacts=()):
        if self.root is None:
            return result
        records = []
        for artifact in artifacts:
            path = Path(artifact).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Cannot checkpoint missing artifact: {path}")
            digest = file_digest(path)
            cached = self.root / "artifacts" / digest if self.root is not None else None
            if cached is not None and not cached.is_file():
                cached.parent.mkdir(parents=True, exist_ok=True)
                temporary = cached.with_suffix(".tmp")
                shutil.copy2(path, temporary)
                os.replace(temporary, cached)
            records.append({"path": str(path), "sha256": digest,
                            "cached_path": str(cached.relative_to(self.root)) if cached else None,
                            "bytes": path.stat().st_size})
        atomic_json(self._path(item_id), {"schema_version": self.schema_version,
            "item_id": str(item_id), "status": "complete", "artifacts": records,
            "result": result})
        return result

    def progress(self, item_ids):
        completed = [str(item_id) for item_id in item_ids if self.read(item_id) is not None]
        pending = [str(item_id) for item_id in item_ids if self.read(item_id) is None]
        return {"fingerprint": self.key, "completed": completed, "pending": pending,
                "completed_count": len(completed), "pending_count": len(pending)}

    def snapshot(self, archive):
        """Atomically replace a portable zip containing this fingerprint tree."""
        if self.root is None:
            return None
        archive = Path(archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary_base = archive.parent / (archive.name + ".building")
        temporary_zip = Path(str(temporary_base) + ".zip")
        if temporary_zip.exists():
            temporary_zip.unlink()
        shutil.make_archive(str(temporary_base), "zip", self.root.parent, self.root.name)
        os.replace(temporary_zip, archive)
        return archive


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


def full_audio_chunks(sample_count, sample_rate, chunk_size=30):
    """Cover every sample with bounded windows; do not infer whether it is speech."""
    if sample_count <= 0 or sample_rate <= 0 or chunk_size <= 0:
        raise ValueError("Audio length, sample rate and chunk size must be positive")
    step = max(1, int(sample_rate * chunk_size))
    return [{"start": left / sample_rate, "end": min(left + step, sample_count) / sample_rate}
            for left in range(0, sample_count, step)]


def create_full_audio_vad():
    """WhisperX coverage adapter for controlled experiments, not a speech detector."""
    from whisperx.vads.vad import Vad

    class FullAudio(Vad):
        @staticmethod
        def preprocess_audio(audio):
            return audio

        def __call__(self, inputs):
            return {"sample_count": len(inputs["waveform"]), "sample_rate": inputs["sample_rate"]}

        @staticmethod
        def merge_chunks(segments, chunk_size, onset, offset):
            return full_audio_chunks(segments["sample_count"], segments["sample_rate"], chunk_size)

    return FullAudio(.5)
