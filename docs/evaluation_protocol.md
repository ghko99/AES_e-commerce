# Evaluation Protocol

Use a consistent protocol when comparing AES-E COMMERCE scoring models.

## Metrics

Track the metrics used by the local scripts, and include at least one agreement metric and one error metric when possible.

- Quadratic weighted kappa.
- Pearson or Spearman correlation.
- Mean absolute error.
- Root mean squared error.

## Result Record

Each result should include the script entrypoint, Git commit, dataset split, preprocessing settings, random seed, and checkpoint path.

## Review

Inspect high-error examples before promoting a metric table. Keep prediction exports outside Git unless they are tiny curated examples.
