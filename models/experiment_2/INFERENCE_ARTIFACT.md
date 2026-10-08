# Tracked localhost inference artifact

The user explicitly approved including the existing
`nev_fault_pipeline.joblib` for the localhost collaboration checkpoint on
2026-10-08. It contains the already-selected Experiment 2 Random Forest
inference pipeline; it was **not retrained, tuned or regenerated** for this
checkpoint. Experiment 1 artifacts and all raw datasets remain excluded.

- Exact size: **2,729,762 bytes** (approximately 2.60 MiB).
- SHA-256: `010ff88628b7d58cb3aa1f8d2430b1087e3aef1936c37082edfde7270acbe987`.
- Saved environment: Python 3.13.9, scikit-learn 1.7.2, pandas 2.3.3,
  NumPy 2.3.5 and joblib 1.5.2.
- Feature order, class mapping, training configuration and unchanged recorded
  results: `metadata.json` in this directory.
- Input scale: seven already-normalized values approximately in [0,1]; original
  physical-unit normalization parameters remain unknown.

The small file is intentionally tracked in normal Git so a collaborator can
pull the application and perform inference without Aarya's local files or
raw training CSVs. `.gitignore` makes an exception for **this exact artifact**,
not every `.joblib` or future model. Earlier experiment documentation describes
the original local-only policy; this explicit checkpoint supersedes that
storage policy without altering the modeling results.

Use Python 3.13 and the pinned inference/application dependencies in
`requirements.txt`. The loader rejects scikit-learn version mismatches rather
than silently running an incompatible artifact. `joblib`/pickle deserialization
can execute code: use only this trusted repository artifact, never an uploaded
or untrusted replacement. No API request controls its path.

The artifact is for the documented proof-of-concept fault-diagnosis module.
Dataset source/license and generation/normalization provenance are still open
items; inclusion for this user-approved collaboration is not a declaration of
unrestricted commercial or third-party redistribution rights. No raw data is
published, and no OEM, temporal or cross-vehicle performance is established.
