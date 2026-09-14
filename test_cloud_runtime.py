import tempfile
import unittest
from types import SimpleNamespace
from cloud_runtime import StageCache, create_face_analyzer, full_audio_chunks, create_full_audio_vad


class CloudRuntimeTests(unittest.TestCase):
    def test_cache_reuses_completed_stage_and_isolates_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = StageCache(directory, {"video": "one", "code": "one"})
            self.assertEqual(cache.get("stage", lambda: {"speaker": "SPEAKER_02"}), {"speaker": "SPEAKER_02"})
            repeated = StageCache(directory, {"code": "one", "video": "one"})
            self.assertEqual(repeated.get("stage", lambda: self.fail("stage reran")), {"speaker": "SPEAKER_02"})
            self.assertIsNone(StageCache(directory, {"video": "two", "code": "one"}).read("stage"))
            self.assertIsNone(StageCache(directory, {"video": "one", "code": "two"}).read("stage"))
            self.assertFalse(list(cache.root.glob("*.tmp")))

    def test_disabled_cache_does_not_save(self):
        cache = StageCache(None, {})
        cache.write("stage", {"ok": True})
        self.assertIsNone(cache.read("stage"))

    def test_face_provider_selection_and_actual_fallback(self):
        for device, available, expected, context in (
            ("cpu", ["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"], -1),
            ("cuda", ["CPUExecutionProvider"], ["CPUExecutionProvider"], -1),
            ("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"], 0)):
            calls = []
            def factory(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(prepare=lambda **options: calls.append(options),
                    models={"recognition": SimpleNamespace(session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))})
            ort = SimpleNamespace(get_available_providers=lambda: available, preload_dlls=lambda: None)
            _, actual = create_face_analyzer(device, ort, factory)
            self.assertEqual(calls[0]["providers"], expected)
            self.assertEqual(calls[1]["ctx_id"], context)
            self.assertEqual(actual["recognition"], ["CPUExecutionProvider"])


    def test_full_coverage_has_no_gaps_and_bounds_final_window(self):
        chunks = full_audio_chunks(65 * 16000, 16000)
        self.assertEqual(chunks, [{"start": 0, "end": 30}, {"start": 30, "end": 60}, {"start": 60, "end": 65}])
        for left, right in zip(chunks, chunks[1:]):
            self.assertEqual(left["end"], right["start"])
        self.assertAlmostEqual(full_audio_chunks(16001, 16000)[-1]["end"], 1.0000625)
        with self.assertRaises(ValueError):
            full_audio_chunks(0, 16000)

    def test_whisperx_adapter_uses_requested_chunk_size(self):
        import numpy as np
        adapter = create_full_audio_vad()
        audio = np.zeros(25 * 16000)
        self.assertIs(adapter.preprocess_audio(audio), audio)
        detected = adapter({"waveform": audio, "sample_rate": 16000})
        self.assertEqual(adapter.merge_chunks(detected, 10, .5, .36),
            [{"start": 0, "end": 10}, {"start": 10, "end": 20}, {"start": 20, "end": 25}])

    def test_pipeline_reuses_stages_voice_and_visual_evidence(self):
        import contextlib
        import io
        import json
        from pathlib import Path
        from unittest.mock import patch, Mock
        import numpy as np
        import torch
        import chainofrules as pipeline
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"video.mp4").write_bytes(b"fake media")
            np.save(root/"voice.npy", np.ones((2, 192), dtype=np.float32))
            np.save(root/"face.npy", np.ones((2, 512), dtype=np.float32))
            records = [{"start": 0., "end": 1., "speaker": "SPEAKER_00"},
                       {"start": 1., "end": 2., "speaker": "SPEAKER_01"}]
            assigned = {"segments": [{"start": 0., "end": 1., "text": "Hello.", "speaker": "SPEAKER_00"}]}
            whisper = Mock(); whisper.transcribe.return_value = {"language": "en", "segments": []}
            voice = Mock(); voice.encode_batch.return_value = torch.ones((1, 1, 192))
            detector = Mock(return_value=pipeline.pd.DataFrame(records))
            cap = Mock(); cap.get.return_value = 30
            argv = ["chainofrules.py", str(root/"video.mp4"), "--voice-priors", str(root/"voice.npy"),
                    "--face-priors", str(root/"face.npy"), "--output", str(root/"result.json"),
                    "--cache-dir", str(root/"cache")]
            with patch("sys.argv", argv), patch.dict("os.environ", {"HF_TOKEN": "test-placeholder"}), \
                 patch.object(pipeline.torch.cuda, "is_available", return_value=False), \
                 patch.object(pipeline.whisperx, "load_audio", return_value=np.zeros(32000)), \
                 patch.object(pipeline.torchaudio, "load", return_value=(torch.zeros(1, 32000), 16000)), \
                 patch.object(pipeline.cv2, "VideoCapture", return_value=cap), \
                 patch.object(pipeline.whisperx, "load_model", return_value=whisper) as load, \
                 patch.object(pipeline.whisperx, "load_align_model", return_value=(Mock(), {})) as align_load, \
                 patch.object(pipeline.whisperx, "align", return_value=assigned), \
                 patch.object(pipeline.whisperx, "assign_word_speakers", return_value=assigned), \
                 patch.object(pipeline, "DiarizationPipeline", return_value=detector) as diarize_load, \
                 patch.object(pipeline.SpeakerRecognition, "from_hparams", return_value=voice) as voice_load, \
                 patch.object(pipeline, "create_face_analyzer", return_value=(Mock(), {})), \
                 patch.object(pipeline, "collect_visual_evidence") as visual, \
                 contextlib.redirect_stdout(io.StringIO()):
                pipeline.main()
                first = json.loads((root/"result.json").read_text())
                pipeline.main()
                second = json.loads((root/"result.json").read_text())
                self.assertEqual(first, second)
                self.assertEqual(load.call_count, 1)
                self.assertEqual(align_load.call_count, 1)
                self.assertEqual(diarize_load.call_count, 1)
                self.assertEqual(voice.encode_batch.call_count, 2)
                self.assertEqual(visual.call_count, 1)
                self.assertEqual(voice_load.call_args.kwargs["run_opts"], {"device": "cpu"})
                self.assertEqual(cap.release.call_count, 2)
                # Coverage changes only ASR/alignment caches, not track identity.
                with patch("sys.argv", argv + ["--transcription-coverage", "full"]):
                    pipeline.main()
                self.assertEqual(load.call_count, 2)
                self.assertIn("vad_model", load.call_args.kwargs)
                self.assertEqual(align_load.call_count, 2)
                self.assertEqual(diarize_load.call_count, 1)
                self.assertEqual(voice.encode_batch.call_count, 2)
                self.assertEqual(visual.call_count, 1)


if __name__ == "__main__":
    unittest.main()
