<p align="center">
  <img src="./assets/header.svg" alt="rppg10 evaluation harness" width="100%">
</p>

# rppg10-eval-harness

A reproducible, offline evaluation harness for comparing camera-derived pulse estimates
with synchronized ECG reference measurements.

Use it when you need a repeatable research pass over rPPG extraction results—not a
medical device, a hosted API, or a source of redistributable health data.

This repository publishes the evaluation method, synthetic tests, and data-rights
boundaries. It does not publish a product lever map, selected operating threshold,
subject-to-health rows, or an implementation roadmap.

## Evidence policy

- Report aggregate error, coverage, uncertainty, and limitations together.
- Treat an inconclusive result as valid.
- Decline to make a performance claim when coverage or reference evidence is inadequate.
- Keep subject-linked artifacts local and out of release archives.
- Do not redistribute datasets, model weights, or derived row-level tables.

## What it evaluates

| Stage | What the harness does | Public by default? |
| --- | --- | --- |
| Ingest | Discovers the expected subject/ROI video layout and records non-identifying run metadata. | Yes |
| Extraction | Runs the configured rPPG method panel through the local rPPG-Toolbox checkout. | Aggregate status only |
| Quality | Computes signal-quality, cross-method, cross-ROI, and confidence fields locally. | Aggregate summaries only |
| Reference scoring | Aligns camera estimates with synchronized ECG reference measurements. | Aggregate metrics only |
| Boundary checks | Rejects subject-linked rows, source videos, arrays, caches, and weights from release archives. | Yes |

The repository intentionally keeps the evidence boundary visible: a passing synthetic
test is not a published dataset result, and an inconclusive result remains inconclusive.

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

The toolbox checkout is an external dependency. Its code and model-weight licence terms
remain in force; this repository does not mirror or relicense it.

## Run

The default command writes only aggregate run counts and non-identifying provenance:

```bash
.venv/bin/python -m rppg10.cli \
  --dataset /path/to/Dataset_rPPG-10 \
  --out ./out
```

The public output is deliberately small:

```text
out/
├── run_meta.json
└── run_summary.json
```

This keeps the default run suitable for sharing without accidentally publishing
subject-linked artifacts. The extractor can also accept a dataset `.zip`; it extracts
to a temporary directory and does not modify the source archive.

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

The suite covers synthetic extraction, ECG alignment, HR/HRV calculations, resume
behavior, worker pickling, JSON sanitization, and release-boundary checks. Dataset-
dependent behavior must be checked locally without publishing source data or subject
mappings.

See [CONTRIBUTING.md](CONTRIBUTING.md) for focused contribution ideas and the privacy
boundary for issues and pull requests.

Most tests use generated signals and synthetic videos. A release-boundary test rejects
row-level tables, source videos, arrays, feature caches, and model weights.

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
