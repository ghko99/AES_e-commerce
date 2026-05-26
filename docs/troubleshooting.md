# Troubleshooting Notes

Use these checks when AES-E COMMERCE runs fail or produce suspicious metrics.

## Data Problems

- Confirm spreadsheet exports were converted with the expected encoding.
- Check that labels remain numeric after preprocessing.
- Verify generated arrays match the source row order.
- Check for overlap between train and test splits.

## Training Problems

- If loss is unstable, inspect label scale and batch shapes first.
- If metrics are unexpectedly high, verify split isolation.
- If results vary across runs, record random seed and package versions.

## Output Problems

Keep prediction files, metric summaries, and logs together so failures can be traced without rerunning the full experiment.
