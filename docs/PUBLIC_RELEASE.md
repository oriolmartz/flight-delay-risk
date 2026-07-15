# Flight Delay Risk v1.5.0 public release

## Deployment checklist

1. Run `python -m scripts.quality_gate`.
2. Build both images with `docker compose build`.
3. Deploy the API and verify `/live`, `/ready` and `/docs`.
4. Deploy the dashboard and run the sample-schedule workflow in English and Spanish.
5. Generate one flight PDF and one schedule PDF in both languages.
6. Add the public dashboard and API URLs to `README.md`, `README_ES.md` and the GitHub repository description.
7. Configure the GitHub social preview with `docs/assets/github_social_preview.png`.
8. Verify that the README shows the business problem, official BTS dataset links and the two real product screenshots.
9. Create the annotated Git tag `v1.5.0` and GitHub Release.


## Suggested release links

```text
Live dashboard: <ADD_AFTER_DEPLOYMENT>
API docs:      <ADD_AFTER_DEPLOYMENT>/docs
Readiness:     <ADD_AFTER_DEPLOYMENT>/ready
```

No hosted URLs are committed until they have been verified. This prevents a public README from presenting dead or fabricated links.

## Runtime notes

- The model artifact is loaded from `models/flightrisk_model.joblib`.
- Prediction logs are written under `monitoring/`.
- Free hosting may introduce cold starts; the UI already distinguishes local benchmark latency from hosted latency.
- The dashboard is self-contained and does not require the API to render because it calls the shared service layer directly.
