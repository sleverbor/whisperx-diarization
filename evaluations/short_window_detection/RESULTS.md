# Short-window speech-detection test

Compared six 20-second community-1 windows with 10-second overlap against a 60-second unconstrained pass and the earlier fixed-three-speaker pass. Same raw audio/model, CPU, unchanged text/reference. Window IDs are not stitched. Comparison region is source 1090–1150 seconds; the last window extends beyond this and is clipped for coverage calculations.

Speech coverage means overlap with saved recovered/aligned utterance spans, not measured transcription or diarization accuracy. Noise and mixed-speaker spans can count as coverage.

| Recovered hypothesis | 60s unconstrained | 20s-window union |
|---|---:|---:|
| My behavior is kind of a reflection of theirs, so I wouldn't agree more. | 100% | 100% |
| I couldn't agree more. | 0% | 0% |
| Couldn't agree a syllable more. | 0% | 0% |
| It's fun to get them off. | 0% | 26% |
| No, that's not the problem. | 0% | 44% |
| They pee. | 0% | 0% |
| It just got stuck. | 82% | 100% |
| I don't want to mess with your cuffs or mess them up or anything, but let's have a look. | 56% | 56% |
| Sounds like a lot of fun. | 0% | 0% |
| We're not trying to tie someone, we're trying to get the key out of them. | 0% | 1% |
| Oh, sorry. | 0% | 0% |
| I might need to get a new key, there might be something off with this one, because I just had an issue with that fella I got in. | 42% | 76% |
| Put this arm on the wall in front of you and that could have cut him off. | 98% | 100% |
| That guy I just searched in that had the shackles, it had a little bit of difficulty opening up one of the shackles. | 100% | 100% |
| I think I got to get a new key then. | 100% | 100% |

## Findings

Shorter windows recover more portions of some unknown/key-officer speech, but do not fix the important missed target agreement or cuff-tightening exchange. The window at source 1120–1140 detects no intervals despite audible speech. Similarity-based speaker identification cannot use an interval that diarization fails to detect.

The known longer target agreement has a direct target voice match from the previous experiment but zero diarization coverage here. Keep that independent evidence; a missing diarization interval should not be interpreted as silence.

Do not adopt window shortening alone as a solution. Next bounded test: inspect or compare speech activity detection itself, with the recovered/audio-supported spans retained for review even when diarization misses them. A detector should also be checked on key-only noise to avoid replacing missed speech with false detections. No identity thresholds or production defaults were changed.
