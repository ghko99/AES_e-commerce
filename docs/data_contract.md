# Data Contract

Document the expected AES-E COMMERCE data shape before running training or evaluation scripts.

## Required Fields

- Essay or review text input.
- Score label or rubric target.
- Split name or fold id.
- Optional prompt, product, or category metadata used by preprocessing.

## Checks

- Confirm labels use the expected numeric range.
- Verify train, validation, and test splits do not overlap.
- Check that generated arrays align with the original row order.
- Keep raw spreadsheets and local exports outside Git.

## Run Metadata

Record the data snapshot, preprocessing command, split seed, and generated artifact paths with every result table.
