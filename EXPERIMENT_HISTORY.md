# Diarization Experiment History

Updated: 2026-09-19  
Project branch: `iterate-local-speaker-evidence`

## Purpose and interpretation

This document records the experiments that produced saved results in this project. It separates measured findings from ideas that were only discussed. Most evaluations use difficult, deliberately selected clips rather than random samples, so percentages describe those review sets and are not general accuracy estimates.

The governing rule that emerged is to preserve the baseline transcript and raw diarization tracks, collect independent evidence, and abstain when the evidence conflicts. Supplemental stages may propose review candidates; they do not silently rewrite words or speaker identity.

## Current state

The working pipeline now provides:

- WhisperX transcription and alignment with pyannote diarization.
- Target mapping led primarily by SpeechBrain ECAPA voice affinity.
- InsightFace observations as supporting evidence when a usable target face is visible.
- Preservation of non-target diarization track IDs rather than forcing every person into a named role.
- Confidence-filtered transcript export with doubtful material retained in a review ledger.
- Targeted recovery from uncovered transcript intervals.
- Independent speech-activity evidence for speech missed by diarization.
- Review-only overlap detection and separation evidence.
- Optional YouTube-caption gap detection with nearby-duplicate suppression.
- GPU use for speech and face models when the runtime supports it.
- Checkpointed Kaggle execution and downloadable result bundles.

The current notebook revision is `caption-gap-review-v28`. Caption text, separated-stream ASR, face presence, semantics, and question/answer context are never sufficient by themselves to assign a speaker or insert words automatically.

## Experiment summary

| Area | Experiment | Main result | Decision |
|---|---|---|---|
| Baseline | Initial 30-second and two-minute tuning | Voice affinity distinguished the target track better than face averages. Short replies remained fragile. | Map the target cluster primarily from voice; retain uncertainty for weak short phrases. |
| Short replies | Temporal question/answer context | Helped tentative attribution of acknowledgements such as brief answers, but could not prove identity from semantics. | Keep as weak evidence only. |
| Reference building | Rebuilt target reference from guaranteed YouTube windows | Produced 18 retained voice samples and 33 face samples. Affinity improved slightly, but the second clip's track-assignment errors remained. | Preserve reproducible builder; do not treat a stronger reference as a cure for bad segmentation. |
| Missing speech | Full-coverage and targeted gap decoding | Recovered the height question/answer, weight exchange, arrest-form dialogue, and handcuff discussion while preserving baseline rows. | Keep gap recovery as review-only additions. |
| Audio preprocessing | Demucs | Did not improve the height/weight transcript and introduced a likely hallucination in the residual. | Do not preprocess the full pipeline with Demucs. |
| Audio preprocessing | Meta DNS64 denoising | A 50% blend was roughly neutral; full denoising seriously damaged ASR and was slower. | Keep original audio as the default. |
| Timing | Independent audio-only timing control | Found late dialogue placed about 12–14 seconds early because omitted intervening speech distorted alignment. | Repair transcription context and timing before tuning identity thresholds. |
| Timing | Complete contextual re-alignment | Corrected four target replies to within 0.40 seconds of the control; all four target voice similarities increased. | Use complete contextual hypotheses as optional repair evidence. |
| Scene identity | Voice-reference matching on corrected crops | Recovered target and known-officer hypotheses, but brief lines, radio, and a third voice remained uncertain. | Useful review evidence; not complete diarization. |
| Speech detection | Short-window diarization | Recovered portions of some turns but still missed important target and officer speech. | Do not use shorter diarization windows alone as the solution. |
| Speech detection | Independent Silero VAD | Recovered several speech spans missed by diarization at the ordinary 0.5 threshold. | Retain speech activity separately; VAD locates speech but does not identify speakers. |
| Local evidence | Sliding voice crops and third-speaker screening | Localized target evidence inside one mixed segment. Candidate third-speaker samples were not consistent enough to promote. | Keep local turn hints; require multiple clean samples for new profiles. |
| Repeated scenes | Waveform-based repeated-presentation matching | Found repeated recordings and attribution conflicts, including replayed opening material. | Use repeats as corroboration evidence without treating duplicates as independent enrollment. |
| Confidence | Conservative transcript export | Allowed doubtful content to be omitted from the readable transcript while retaining a review ledger. | Integrated as reversible presentation policy. |
| Overlap detection | DiaPer and Sortformer comparison | Sortformer was the better primary detector; DiaPer added some rescue evidence but also false triggers. | Adopt additive two-tier review policy; neither model rewrites identity. |
| Separation | Contextual WeSep | Two of five clips yielded useful target streams; longer context did not help. | Review-only; use three seconds if retained. |
| Separation | SepFormer blind two-output | Two of five clips yielded useful target streams, partly different from WeSep. ECAPA chose the wrong stream for both confirmed recoveries. | Keep both streams for review; do not trust short-crop ECAPA selection. |
| Separation | Multi-offset SepFormer | Target-dominant audio sometimes repeated across offsets, but words were unstable. | Useful existence evidence, not automatic wording. |
| Separation | Context-aligned separated ASR | Only 2/9 aligned readings were correct; timestamps on separated audio were unstable. | Do not insert words from separated-stream timestamps. |
| Separation | MossFormer2 benchmark | Found target-containing output in 4/4 target-present clips versus SepFormer 2/4; two outputs were clean target-only. | Best tested separator, but still review-only. |
| Separation | MossFormer2 full priority review | 6/15 outputs were useful, 6 were wrong/garbled, 3 were not the target; only one added missing speech. | Voice scores triage audio but do not validate words. |
| ASR corroboration | Mixture versus separated-stream decoding | Strong agreement was useful in only 2/5 unprompted cases; context prompting increased false agreement. | Reject same-model cross-audio agreement as an automatic rule. |
| Captions | Captions versus 15 overlap clips | On all six useful MossFormer2 clips, captions favored the baseline or were inconclusive; none favored the separated candidate. | Captions help wording/timing, not speaker identity. |
| Captions | Full-timeline missing-speech detection | Two of four candidates were genuine omissions. Both false candidates duplicated nearby transcript wording. | Integrated duplicate suppression; export remaining gaps for review. |

## Detailed findings

### 1. Baseline and short-phrase attribution

The early 30-second tests established that voice affinity was the strongest identity signal available. In the representative opening result, the target cluster affinity was about 0.262 versus 0.060 for the other track, while face averages were nearly tied. Face evidence therefore became supporting evidence rather than part of the identity lock.

Temporal conversation evidence improved some weak acknowledgements and answers, leading to the `milestone-two-minute-diarization` tag. The rule remains tentative: a reply following a question can support a hypothesis, but police-encounter semantics do not prove that a particular person answered.

### 2. Target reference experiments

The source-based reference builder uses known target-only YouTube windows and preserves provenance. The evaluated candidate contained 18 voice samples and 33 face samples. Opening target affinity changed from 0.255 to 0.257; the second clip changed from 0.439 to 0.455. The second clip still received no final target labels because target speech had been assigned to the wrong diarization track. This demonstrated that enrollment quality and track assignment are separate problems.

The later reference-promotion workflow exports only high-confidence current-video samples for human approval. Promoted samples remain provenance-tracked and are added after processing rather than allowing a video to train on its own uncertain decisions during the same run.

Sources: [reference comparison](evaluations/reference_comparison/RESULTS.md), [hour iteration](evaluations/hour_iteration/RESULTS.md).

### 3. Missing transcription and noisy speech

Targeted gap decoding recovered important dialogue that normal coverage skipped. A long isolated window recovered “How tall are you?”, the height answer, both weight questions, “Sorry?”, and “160.” Adding surrounding audio improved the weight question compared with a bare gap crop. A second interval recovered the arrest-form explanation and a short response exchange.

All baseline segments were preserved byte-for-byte. Recovered phrases were stored separately because noisy windows also produced questionable radio wording and incomplete fragments.

Demucs did not improve the tested height/weight interval. The vocals track changed a likely-correct height from 5'10 to 5'9, while the residual hallucinated an unsupported outro. Meta DNS64 full denoising changed coherent questions into incoherent phrases and took roughly five times longer than the dry or lightly blended audio. The useful gain came from choosing a bounded interval with enough context, not from aggressive preprocessing.

Sources: [gap recovery](evaluations/gap_recovery/RESULTS.md), [context windows](evaluations/gap_context/RESULTS.md), [Demucs](evaluations/demucs_problem_area/RESULTS.md), [denoising](evaluations/denoiser_test/RESULTS.md).

### 4. Timing, alignment, and scene-local identity

A saved late-video transcript placed several replies 12–14 seconds early because the short hypothesis omitted officer and radio speech. An independent audio-only decode exposed the error. Re-aligning the complete 25-second hypothesis moved four known target responses to within 0.40 seconds of the control and increased every target voice score; the strongest changed from -0.045 to 0.516.

With corrected crops, fixed voice gates produced three target hypotheses, three known-officer hypotheses, and seven uncertain lines. Radio speech and very short words remained unresolved. This established that correct timing materially improves voice evidence, while also showing that crop repair does not reconstruct a missing diarization speaker track.

Sources: [timing diagnostic](evaluations/timing_diagnostic/RESULTS.md), [context alignment](evaluations/context_alignment/RESULTS.md), [speaker separation](evaluations/speaker_separation/RESULTS.md).

### 5. Speech activity and turn detection

Short overlapping diarization windows improved coverage for some key-officer speech but still missed the longer target agreement and other audible turns. Independent Silero activity at its standard 0.5 threshold recovered many of those spans, including target agreement and an officer response. It also correctly rejected digital silence in the tested control.

Speech activity is now treated as independent evidence. A segment can contain speech even when diarization has no interval. VAD cannot identify the speaker, repair words, or separate simultaneous voices.

Sliding 1.2-second voice crops localized target evidence inside one mixed reflection/agreement sentence. Attempts to build a new anonymous third-speaker profile failed consistency screening, so no profile was promoted.

Sources: [short-window detection](evaluations/short_window_detection/RESULTS.md), [independent speech detection](evaluations/speech_detection/RESULTS.md), [local evidence](evaluations/local_evidence_pass/RESULTS.md).

### 6. Overlap detection

A 35-clip hand-labeled diagnostic set compared baseline, Sortformer, and DiaPer. After excluding two unclear clips, 21 contained intelligible simultaneous speech and 12 were rapid turn boundaries.

- Sortformer detected 17/21 overlaps with any positive overlap.
- DiaPer detected 14/21.
- Sortformer overlap fraction above 0.30 produced 0.80 precision and 0.57 recall.
- A permissive Sortformer-plus-DiaPer rescue detected 19/21 but caused 10 false triggers among 12 boundary clips.

The implemented policy is additive: strong Sortformer detections go to separation/review; weaker or DiaPer-only evidence raises uncertainty. It never replaces the baseline or directly changes identity.

Source: [hand-labeled overlap evaluation](evaluations/overlap-hand-label-v1/README.md).

### 7. Speaker separation

Five fixed overlap exchanges were used across methods.

**Contextual WeSep:** Three and five seconds of context each produced two useful target streams, two mixed streams, and one failure/unclear result. Five seconds reduced mean target-word F1 from 0.477 to 0.339.

**SepFormer two-output separation:** Two clips contained useful target streams, but ECAPA selected the wrong stream for both. Short separated crops distorted identity embeddings.

**Multi-offset separation:** Human listeners repeatedly found a target-dominant stream at different offsets, but the decoded words varied and sometimes hallucinated. Full-context separated-stream decoding with timestamp alignment yielded only 2 correct, 4 incorrect, and 3 unclear readings.

**MossFormer2:** On the five-clip benchmark, it produced target-containing audio in all four target-present clips and clean target-only audio in two. SepFormer had produced target-containing audio in two of four. In the larger 15-clip priority review, MossFormer2 produced six useful target-bearing outputs, but six were wrong or garbled and three were not the target. Only one added missing speech. There were no safe automatic transcript insertions.

MossFormer2 remains the strongest tested separator, used only to produce review audio. Voice similarity and margin select likely target-bearing audio; they do not validate the words produced by ASR.

Sources: [contextual WeSep](evaluations/contextual-wesep-v1/RESULTS.md), [two-output separation](evaluations/two-output-separation-v1/RESULTS.md), [multi-offset separation](evaluations/multi-offset-separation-v1/RESULTS.md), [context-aligned ASR](evaluations/context-aligned-separated-asr-v1/RESULTS.md), [MossFormer2 benchmark](evaluations/mossformer2-separation-v1/summary.json), [MossFormer2 priority review](evaluations/mossformer2-full-priority-v1/summary.json).

### 8. Attempts to validate separated words

The original mixture and target-like separated stream were decoded independently with the same ASR model. Strong unprompted agreement occurred five times, but only two were useful; three repeated wrong words or the wrong speaker. Supplying surrounding transcript context produced six strong agreements, four of which were non-useful. The two audio versions can preserve the same interference, so the model can confidently repeat the same error.

A larger-model Kaggle run was not justified by this screen. Same-model agreement is not independent corroboration.

Source: [cross-audio ASR result](evaluations/mossformer2-corroborated-asr-v1/RESULTS.md).

### 9. YouTube captions

Captions were evaluated as independent wording and timing evidence. Across the 15 MossFormer2 review clips, captions supported the baseline more strongly in seven, the separated candidate in four, and both similarly in three. For all six human-useful MossFormer2 clips, captions favored the baseline or were inconclusive. Caption agreement with separated text sometimes confirmed real words spoken by the wrong person.

The full-timeline gap test initially produced four candidates. Human review found two genuine omissions: one non-target interval and one multi-speaker overlap. The other two were music or delayed duplicate captions. Both false candidates repeated nearby baseline wording, so duplicate suppression reduces this calibration set from four candidates to the two genuine omissions.

The reviewer also heard distant speech under music that was absent from both the baseline and captions. Captions therefore improve gap discovery but cannot detect content omitted by both systems.

The full pipeline now fetches or accepts automatic captions, identifies uncovered caption timing, suppresses nearby wording duplicates, and exports short review clips. Caption text never supplies identity or automatic transcript corrections.

Sources: [caption wording comparison](evaluations/youtube-caption-priority-v1/RESULTS.md), [caption gap review](evaluations/caption-gap-review-v1/RESULTS.md).

### 10. Visual evidence

Higher-resolution face processing and spatial tracking improved continuity when faces were visible. Face presence remained insufficient for active-speaker attribution: heads were frequently rotated or cropped, mouths were obscured, and visible mouth movement could occur while another person spoke. The system does not claim lipreading. Visual evidence supports identity only when a usable recognized face is present; it does not override voice evidence or establish that the visible person is speaking.

### 11. Cloud and reproducibility work

The Kaggle workflow was stabilized through an isolated virtual environment, explicit CUDA checks, GPU ONNX Runtime, repaired WeSep packaging, H.264 video normalization, staged checkpoints, per-video result keys, early credential loading, and local download artifacts. On T4×2 runs, MossFormer2 uses GPU 0 while ECAPA and Whisper use GPU 1 to avoid memory exhaustion.

Milestones:

- `milestone-two-minute-diarization`: solid short-clip baseline with tentative short-answer handling.
- `milestone-full-video-audiovisual-v8`: first stable full-video audiovisual run with target speech recovered across fragmented tracks.
- Current branch head includes MossFormer2 review policy, caption gap detection, and `caption-gap-review-v28` notebook integration.

## Methods rejected as automatic rules

The following were tested and should not be reintroduced without new evidence:

- Uniform or visible officer implies the officer is speaking.
- Face presence overrides voice identity.
- Question/answer semantics alone determine the respondent.
- A stronger target reference repairs diarization-track mistakes.
- Demucs or full denoising improves every noisy interval.
- Short diarization windows reliably recover missed turns.
- A forced fixed speaker count produces clean identities.
- Short-crop ECAPA reliably selects a separated target stream.
- Repeated separation across offsets validates exact words.
- Whisper timestamps on separated audio safely map words back to source time.
- Agreement between mixture ASR and separated-stream ASR proves the words.
- Caption agreement proves target attribution.
- High heuristic strength is a calibrated accuracy probability.

## Recommended next validation

Run the current full pipeline on another video by the same target with automatic captions available. Evaluate separately:

1. Baseline target attribution on ordinary single-speaker turns.
2. Caption-gap precision after duplicate suppression.
3. MossFormer2 review yield on independently chosen overlaps.
4. Reference promotion candidates after the run has completed.

Do not tune on that video before recording the initial results. Reserve reviewed clips from later videos as a holdout set so future threshold changes can be checked for regressions.
