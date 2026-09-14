# Independent speech detection test

Same raw handcuff audio, source 1090–1150 seconds. Compared Silero from faster-whisper at thresholds 0.5, 0.35 and 0.2 with community-1 scene diarization. No speech padding, 100ms minimum speech, 200ms minimum silence. Text, references and original diarization were not modified.

Coverage is interval overlap against saved recovered/aligned hypotheses, not transcription accuracy or manually annotated speech recall. Some hypotheses are garbled or mixed-speaker.

| Recovered hypothesis | Diarization | Silero 0.5 |
|---|---:|---:|
| My behavior is kind of a reflection of theirs, so I wouldn't agree more. | 100% | 100% |
| I couldn't agree more. | 0% | 100% |
| Couldn't agree a syllable more. | 0% | 92% |
| It's fun to get them off. | 0% | 72% |
| No, that's not the problem. | 0% | 100% |
| They pee. | 0% | 97% |
| It just got stuck. | 82% | 100% |
| I don't want to mess with your cuffs or mess them up or anything, but let's have a look. | 56% | 45% |
| Sounds like a lot of fun. | 0% | 0% |
| We're not trying to tie someone, we're trying to get the key out of them. | 0% | 100% |
| Oh, sorry. | 0% | 72% |
| I might need to get a new key, there might be something off with this one, because I just had an issue with that fella I got in. | 42% | 96% |
| Put this arm on the wall in front of you and that could have cut him off. | 98% | 98% |
| That guy I just searched in that had the shackles, it had a little bit of difficulty opening up one of the shackles. | 100% | 100% |
| I think I got to get a new key then. | 100% | 100% |

## Outcome

Silero at its ordinary 0.5 gate recovers the key missed agreement and original-officer response spans. There is no need to lower the gate to achieve those particular gains. Some portions of the officer cuff-inspection offer remain undetected; coverage is not universally better. Lower gates also include more of a hypothesis that failed alignment, so added coverage alone is not a reason to choose them.

All thresholds rejected the five-second digital-silence control. The source 1100–1102 listening candidate also receives zero speech coverage, but has not been independently confirmed as key-only. No measured key-noise false-positive rate is claimed. Its listening copy is attenuated for playback only; detection used unchanged float audio.

## Implementation implication

Treat independent speech activity, aligned transcript evidence and diarization as separate inputs. If audio-supported text has a strong voice match but no diarization interval, retain it as a reviewed voice hypothesis rather than discard it as silence. Silero intervals locate speech; they do not identify people, repair wording, split simultaneous speakers or establish lipreading.

Next integration can add Silero activity as evidence for local recovery candidates, preserve original track IDs and require clean multi-sample voice agreement for an unnamed third speaker. Mixed-speaker blocks still need separation before identity decisions. Existing files and production defaults remain unchanged.
