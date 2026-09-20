# nomo-pVAD experiment setup

The evaluation intentionally keeps downloaded code and model files out of Git.
Recreate the tested dependency with:

```bash
git clone --depth 1 --branch release-1.1 https://github.com/tuya/nomo-pvad.git vendor/nomo-pvad
.venv/bin/python -m pip install "modelscope>=1.10" addict
```

The first inference downloads the ERes2NetV2 enrollment model. Set writable
`MODELSCOPE_HOME` and `MODELSCOPE_CACHE` directories, then pass the resulting
model directory to `evaluate_nomo_pvad.py --eres2netv2-dir`. The experiment uses
`references/auditor-reviewed-v14/auditor_enrollment.wav`; ECAPA embedding arrays
are not compatible with nomo-pVAD's ERes2NetV2 enrollment vector.

The exact benchmark cases are in `evaluations/nomo-pvad-v1/manifest.json`.
