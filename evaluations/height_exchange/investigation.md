# Missing-speech investigation

Input: original video.mp4, source 640–665 seconds (0:15–0:40 of the second test clip), decoded to 16 kHz mono. No topic prompt was supplied.

## Confirmed coverage failure

The saved WhisperX ASR input chunks in the 65-second run were 0.048–16.703 and 40.193–62.317 seconds. Speech in between was excluded before decoding. The user confirms a height question and answer during the noisy entrance. Transcription recovery is tested separately from identity attribution.

## Raw versus cleaned direct transcription

The same large-v2 model ran on CPU/int8 with Faster-Whisper voice filtering disabled. Raw audio recovered:

- source 651.88–652.58: “How tall are you?”
- source 652.86–654.66: “5'10 and a half, 5'11.”
- source 661.42–663.74: a weight question, “Sorry?”, and the repeated question.
- source 663.84–664.28: “160.”

These are model hypotheses requiring listening/ground-truth review, not confirmed wording. Another phrase around 655–657 seconds appears implausible and must remain unresolved.

The cleaned comparison used highpass 80 Hz, lowpass 7000 Hz, and FFmpeg afftdn nr=6/nf=-35. It retained the height exchange but garbled several other phrases and introduced additional questionable text. Cleaning is not selected as a default.

## Face and identity findings

Cached high-affinity face detections in the first seven utterances are near x=950–1040, y=348–448 and align with the actual head in a viewed frame. The shirt graphic is lower in the image. There is no evidence that shirt-face confusion caused these early attribution failures; this check does not verify every frame or active speaker.

The isolated diarizer assigns all aligned utterances to SPEAKER_00 despite detecting two target-track intervals. Only one SPEAKER_01 interval is long enough for a voice signature, so the current requirement for two independently sampled track profiles blocks correction of contradictory local voice evidence. The full-video comparison retains global tracks for future tuning.

No production transcription or identity rules have been changed in this investigation.

## Original-decoder control

A final pass used the original WhisperX large-v2 CPU/int8 batched decoder (batch size 16), with a manually assigned coverage VAD that includes this entire 25-second interval. It recovered both height and weight exchanges from raw audio. `whisperx_full_interval.json` records that result. The same implausible phrase remains in the middle, so forced coverage does not make all wording trustworthy. This control supports fixing VAD coverage before changing recognizer or denoising models.
