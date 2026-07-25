<p align="center">
  <img src="./assets/header.svg" alt="rppg10 evaluation harness" width="100%">
</p>

# rppg10-eval-harness

A reproducible, offline evaluation harness for comparing camera-derived pulse estimates
with synchronized ECG reference measurements.

This repository publishes the evaluation method, synthetic tests, and data-rights
boundaries. It does not publish a product lever map, selected operating threshold,
subject-to-health rows, or an implementation roadmap.

## Evidence policy

- Report aggregate error, coverage, uncertainty, and limitations together.
- Treat an inconclusive result as valid.
- Decline to make a performance claim when coverage or reference evidence is inadequate.
- Keep subject-linked artifacts local and out of release archives.
- Do not redistribute datasets, model weights, or derived row-level tables.

## Setup

Python 3.12 is recommended.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .
```

The unsupervised extraction methods are reused from
[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox). Point the harness at a
compatible local checkout:

```bash
export RPPG_TOOLBOX_PATH=/path/to/rppg-toolbox
```

## Run

The default command writes only aggregate run counts and non-identifying provenance:

```bash
.venv/bin/python -m rppg10.cli \
  --dataset /path/to/Dataset_rPPG-10 \
  --out ./out
```

Detailed waveforms, timelines, manifests, and subject-linked rows are local research
artifacts. Generate them only with explicit private-output mode:

```bash
.venv/bin/python -m rppg10.cli \
  --dataset /path/to/Dataset_rPPG-10 \
  --out ./private-out \
  --private-row-output
```

Never commit or publish that output directory.

Aggregate scoring requires a private extraction run because the raw subject association
is needed locally to pair camera estimates with ECG:

```bash
python -c "from rppg10.cli import main_score; \
main_score(['--dataset','/path/to/Dataset_rPPG-10', \
'--out','./private-out'])"
```

The scorer writes aggregate ROI, skin-tone, and overall files. A subject-linked clip file
is produced only when `--private-row-output` is supplied to the scoring command.

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Most tests use generated signals and synthetic videos. Dataset-dependent behavior must be
checked locally without publishing source data or subject mappings. A release-boundary
test rejects row-level tables, source videos, arrays, feature caches, and model weights.

## Data and model rights

This repository does not include or grant rights to Dataset_rPPG-10, rPPG-Toolbox model
weights, or any other external dataset or model. Obtain each dependency from its owner and
follow its licence and access terms. Dataset_rPPG-10 is distributed separately under
CC BY-NC-SA 4.0 terms.

The Apache-2.0 licence applies only to this repository's code. It does not extend to
external data, weights, or locally derived subject-linked artifacts.

## Public-data boundary

Before release, verify the complete tree and archive. Do not publish subject identifiers,
health attributes, demographic rows, private paths, raw data, weights, feature caches,
selected thresholds, comparative lever results, or unpublished next experiments.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
