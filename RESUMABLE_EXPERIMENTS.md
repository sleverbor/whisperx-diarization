# Resumable model experiments

Large Kaggle experiments must checkpoint each independent item rather than only
writing one report at the end. `cloud_runtime.ResumableWorkSet` provides the
shared implementation.

Each experiment fingerprint includes its input files, model weights, code or
schema revision, and settings that affect output. A changed fingerprint creates
a separate cache tree. Each completed item is written atomically; interrupted
`.tmp` files are ignored. Declared output artifacts are reusable only while
their SHA-256 digest still matches the checkpoint. Artifacts are also stored
once by content digest inside the cache. A restored portable snapshot repairs
missing output artifacts before returning the completed item.

Experiments may periodically call `snapshot()` to atomically refresh a portable
zip. On Kaggle, attach the latest checkpoint zip or its expanded dataset to a
new session and restore it into the configured cache directory. Valid completed
items are reused; missing, partial, corrupt, or configuration-mismatched items
run again.

The nomo-pVAD evaluator is the reference implementation. Its optional arguments
are:

```text
--cache-dir /kaggle/working/stage-cache/nomo-pvad
--snapshot-archive /kaggle/working/nomo-pvad-checkpoints.zip
--snapshot-every 5
```

Upcoming audiovisual extraction and target-speaker ASR notebooks should use the
same pattern with one item per bounded video window. Expensive stages inside an
item may use separate item IDs such as `window-004/separation` and
`window-004/asr`. Final reports are rebuilt from validated item checkpoints and
written atomically; they are never themselves the sole record of progress.

The full Kaggle notebook uses this pattern for every overlap-extraction window
and for the separation, voice-scoring, and ASR phases of every MossFormer2
window. It refreshes dedicated portable ZIPs after every five newly completed
items in addition to the complete stage-cache archive.
