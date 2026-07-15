# UI metric guide

Flight Delay Risk is designed to be understandable without reading the repository documentation.

## Visible decision metrics

| UI label | Meaning | Direction |
|---|---|---|
| Estimated delay risk | Estimated probability that arrival is at least 15 minutes late. | Context-dependent |
| Risk compared with this route | Current estimate divided by the route's historical delay rate. | Above 1× means higher than usual |
| Evidence behind the route baseline | Number of earlier route observations used to form the comparison. | More evidence is generally better |
| Ranking quality | PR-AUC translated into product language: how well delayed flights rise toward the top. | Higher is better |
| Priority-list advantage | Lift at the top 10%: improvement over reviewing flights at random. | Higher is better |
| Probability reliability gap | Average gap between predicted risk and observed outcomes. | Lower is better |

## Progressive disclosure

The default view answers four questions:

1. What is the estimated risk?
2. Is it unusual for this route?
3. How much historical evidence supports the comparison?
4. What factors moved the estimate?

Raw model scores, calibrator names, Brier score, ECE, full fold diagnostics and runtime metrics are kept in **Advanced** expanders.

## Interpretation guardrail

Model contributions describe associations learned by the model. They do not establish real-world causes of delay.
