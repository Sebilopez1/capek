# Security

## Loading checkpoints

Treat a model checkpoint like code: only load checkpoints you trained yourself or got from a source you trust.

- **`bc:<dir>`** (checkpoints from `capek train-bc`): the weights are loaded with `torch.load(..., weights_only=True)`,
  which refuses arbitrary pickled objects. The weights hash is checked against `capek_policy.json` before use, and a
  checkpoint whose architecture doesn't match the simulator's inputs and outputs is refused.
- **`lerobot:<dir>`** (LeRobot `pretrained_model` directories): these are loaded by LeRobot's own code
  (`from_pretrained` plus the saved pre/post-processors), and Capek doesn't sandbox it. Its safety is
  LeRobot's and PyTorch's. Only load LeRobot checkpoints you trust.
- `capek score` only reads parquet files and JSON metadata. It never executes code from a dataset and never writes into it.

Everything runs locally. Capek makes no network calls of its own, and uploads and sends no telemetry.

## Reporting a vulnerability

Please don't open a public issue for security problems. Use GitHub's private vulnerability reporting
("Report a vulnerability" under the repository's Security tab:
https://github.com/Sebilopez1/capek/security). We'll reply as soon as we can. This is a small,
volunteer-run project with no guaranteed response time.

## Supported versions

Only the latest release gets fixes.
