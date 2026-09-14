The speech denoiser did not improve transcription in this problem area. Keep the original audio as preferred. A 50% denoised blend retained the important words, but full denoising severely degraded ASR.

| Item | Original | 50% denoised blend | Full denoising |
|---|---|---|---|
| Height question | How tall are you? | How tall are you? | That's all of you |
| Height answer | 5'10 and a half, 5'11 | 5'10 and a half, 5'11 | It's not 10 minutes, it's not 11 |
| Weight question | How much do you weigh? | How much do you weigh? | I'm a few in |
| Weight response | Sorry? … 160 | Sorry? … 160 | Sorry? … 160 |

The light blend produced essentially the original transcript, with minor wording changes in the tattoo discussion and subsequent height wording. It avoided the height-number mismatch introduced by the preceding Demucs music-separation test. These results do not establish that the light blend is more accurate: there is no complete human-verified reference transcript. Exact wording still requires listening.

Full denoising produced many incoherent substitutions and extra words. Its ASR run took about 232 seconds versus 48 seconds for original and 47 seconds for the light blend, excluding model initialization. More aggressive processing was harmful to recognition in this example.

The test used Meta DNS64, not the music-separation htdemucs model. The denoiser package was installed into an isolated workspace folder with no dependency changes to the existing project environment. No transcription or speaker matching defaults were changed. The input was the same 25-second recording, source 10:40–11:05, as the preceding Demucs comparison; all three ASR configurations were identical.

Signal checks confirmed identical original samples, equal 400,000-sample lengths, finite outputs, the exact 50% dry/wet mix, and reconstruction of original from full output plus removed residual. FLOAT saving avoided clipping and individual loudness normalization. Model and input hashes are recorded.

For listening, audio/listen_original.wav, audio/listen_light.wav, audio/listen_full.wav and audio/listen_removed.wav share one common attenuation to avoid playback clipping. Their relative levels are preserved. This attenuation was applied only to listening copies; the ASR inputs are unchanged and saved separately as *_asr.wav. Listening gain is recorded in audio/listening_summary.json.

The current evidence favors the original audio with targeted gap decoding and enough surrounding context, rather than automatic denoising of the entire video. This single test does not rule out other denoisers or other problem intervals.
