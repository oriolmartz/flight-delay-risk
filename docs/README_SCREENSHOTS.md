# README screenshot set

The current release screenshot set was captured from the redesigned dashboard on 2026-07-22. Reuse these settings when a future UI release requires replacements.

## Capture settings

- Browser width: 1,440–1,600 px.
- Browser zoom: 90%.
- Language: English for `README.md`; the interface is visibly bilingual through the language selector, so a duplicate Spanish set is unnecessary.
- Format: PNG, no browser chrome, no desktop background and no personal information.
- Keep one consistent viewport and crop style across the set.
- Use the bundled demo values where possible so the README and UI tell the same story.

## Required captures

| File | Capture | Must be visible |
|---|---|---|
| `docs/assets/readme_landing.png` | Top of the application | Source/row-count strip, compact header, main statement, U.S. coverage map and demo risk card. **Completed.** |
| `docs/assets/readme_analyze.png` | Analyze flight after submitting the demo | Calibrated probability, queue status, route baseline, support and historical evidence. **Completed.** |
| `docs/assets/readme_heatmap.png` | Heatmap under Analyze flight | Full map, airport labels, artifact-backed points and legend. **Completed.** |
| `docs/assets/readme_rank.png` | Rank schedule after loading the sample CSV | Ordered queue, probabilities, route rates, support, distribution and diagnostics. **Completed.** |
| `docs/assets/readme_validation.png` | Validation | Final-test metric cards and the plain-language metric guide. **Completed.** |
| `docs/assets/readme_model_comparison.png` | Model comparison | Seven-candidate chronological selection table. **Completed.** |

## README placement

1. Replace the top `readme_hero.png` with `readme_landing.png`.
2. Place `readme_analyze.png` after **Analyze one flight**.
3. Replace the temporary static heatmap illustration with `readme_heatmap.png` in **Explore airport history**.
4. Place `readme_rank.png` after **Rank an entire schedule**.
5. Place `readme_validation.png` after **Inspect validation and operations**.
6. Keep `readme_model_comparison.png` in **Model comparison** and label it as selection evidence, not final-test performance.

Keep captions short and decision-oriented. The screenshots should prove the product surfaces; the text below them explains the technical contract.
