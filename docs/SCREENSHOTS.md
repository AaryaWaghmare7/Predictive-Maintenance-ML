# EVision — Screenshot and GitHub Presentation Checklist

## Capture provenance

Browser screenshot tools were unavailable during the 3 November 2026 release
pass. No new UI screenshot was fabricated or claimed.

- `reports/figures/showcase/dashboard.png`: genuine single-logo homepage capture
  from the prior local browser verification on 1 November 2026. It replaces the
  older duplicate-logo capture. Converted to real PNG without redrawing the UI.
- `reports/figures/showcase/telemetry.png`: earlier real four-row CSV sample
  analysis. Converted from JPEG bytes to real PNG; UI content is unchanged.
  The README labels it as an earlier preview requiring a fresh capture.
- `reports/figures/experiment_2/test_confusion_matrices.png`: original genuine
  research figure, already PNG. Preserved unchanged.

The current application still uses the earlier dashboard wording. The broader
UI/UX improvement plan was audited but not approved for implementation; do not
claim those proposed refinements shipped in this presentation checkpoint.

## Fresh captures to make before LinkedIn publication

Start the app from the repository root:

```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8012
```

Use a clean desktop browser viewport, approximately 1366 × 900 CSS pixels.
Hide developer tools and ensure no API token or private information is visible.
Wait for the entrance animation to finish before taking a screenshot.

1. **Homepage:** open `http://127.0.0.1:8012/#home`. Capture the header with its
   ECG/EVision logo, the tagline, headline, vehicle visual, and action buttons.
   The hero should not repeat the wordmark or brand subtitle. Save an actual
   PNG to `reports/figures/showcase/dashboard.png`.
2. **Sample analysis:** click **Download Sample CSV**, then upload that same
   `synthetic_normalized_telemetry.csv` and select **Analyze Telemetry**. Expect
   four observations, one model prediction per class, each 25%. Capture the
   distribution and expanded row results. Save an actual PNG to
   `reports/figures/showcase/telemetry.png`. In a post, caption it
   **“Included four-row sample — model predictions, not live vehicle telemetry.”**
3. **Model evaluation:** use the existing
   `reports/figures/experiment_2/test_confusion_matrices.png`. Caption it
   **“Supplied NEV held-out test results; rows actual, columns predicted.
   Dataset performance does not establish real-world fleet/OEM validity.”**

Inspect all three images at full size. Verify PNG file signatures rather than
changing only a filename extension. After fresh captures, update the README's
capture dates and remove the earlier-preview qualification only when warranted.
Do not paste the reference mockup, invent telemetry, or edit metrics into a capture.

## GitHub About settings

Prepared description:

> EVision is a research prototype for EV sensor-based fault diagnosis using a Random Forest model, FastAPI and an interactive dashboard.

Suggested topics:

`electric-vehicles` · `machine-learning` · `fault-diagnosis` ·
`predictive-maintenance` · `python` · `fastapi` · `scikit-learn`

Set these through the repository's **About** settings. The repository slug remains
`Predictive-Maintenance-ML`. No repository rename or deployment is part of this
checkpoint. Review dataset/artifact redistribution rights before a wider public
research launch; the high test scores do not resolve that permission question.
