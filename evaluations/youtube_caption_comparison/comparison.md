# YouTube caption comparison

Source: https://www.youtube.com/watch?v=jKiST2J1pBg

English captions are auto-generated; no creator-edited English track was available. This test compares captions against existing/fresh audio-only results. Captions and the user's listening notes were not supplied to the ASR model. No caption-prompted decode was performed and no transcript corrections were automatically applied.

## Height/weight, approximately 10:40–11:05

Captions contain both questions, but flatten the height measurements into ambiguous digit strings and merge the repeated weight question with the answer. Our contextual audio-only pass gives more useful height wording and retains the weight answer. Caption timing is useful for identifying speech that the original baseline skipped; caption wording is not consistently better.

## Handcuffs, approximately 18:12–19:02

Captions cover most of the missing conversation, including the stuck-key explanation, cuff-adjustment request, tightness concern and replacement-key discussion. Compared with the user's separately saved listening notes, captions capture the key/tighten concepts that audio-only windows garble. Audio-only windows recover the fuller agreement response that captions simplify. Both contain doubtful muttering/background speech. No reliable radio decoding or speaker identification was established.

The baseline has a long untranscribed interval, not silence: caption timestamps provide useful speech-location evidence. Caption formatting merges some apparent speaker turns; caption chevrons are not stable speaker IDs. Neither a visible person nor a caption line establishes identity.

## Practical next step

Use timed captions to flag uncovered speech intervals and show alternative wording in a review pane beside the original audio. Keep the original audio-only transcript, uncertainty and candidate provenance. Do not automatically replace words or derive speaker identities from captions. A separate blinded caption-assisted decoding experiment would be required to measure whether prompting improves accuracy without copying suggestions.

Full downloaded captions are retained only as local intermediate material, rather than distributed with this comparison. Re-run downloads using yt-dlp with skip-download, write-auto-subs, English original language and json3 format.
