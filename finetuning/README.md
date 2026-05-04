# Fine-tuning experiments

Fine-tunes XLM-R-base and AfroXLMR-base on the
[MasakhaNEWS](https://huggingface.co/datasets/masakhane/masakhanews)
classification benchmark for four agglutinative Bantu languages
(Luganda, Rundi, chiShona, Kiswahili) at controlled training-set sizes.

The companion N-gram baselines and aggregation logic live at
[SACA-Lab/Ngram](https://github.com/SACA-Lab/Ngram). The two repositories
share the same dataset preprocessing, label set, seeds, and held-out splits
to enable directly comparable results and joint analysis.

## Repository layout

```
configs/
    base.yaml              shared training configuration
    xlm-r.yaml             XLM-R-base override
    afroxlmr.yaml          AfroXLMR-base override
src/
    data.py                MasakhaNEWS loader and stratified subsampler
    train.py               single-trial training entry point
    predict.py             test-set inference and parquet export
    metrics.py             weighted F1 and paired t-test
scripts/
    smoke_test.sh          single end-to-end run for pipeline validation
    check_class_coverage.py per-split and per-subsample class counts
    run_grid.sh            full grid runner; resumable
    aggregate.py           text summary tables from results/metrics.csv
    figures.py             publication figures (PDF and PNG)
    tables.py              LaTeX tables
colab/
    finetune_colab.ipynb   Colab notebook
    build_notebook.py      regenerates the notebook
results/
    predictions/{run_id}.parquet
    logs/{run_id}.json
    checkpoints/           cleared after each run
    metrics.csv            appended per run
    figures/               figures emitted by scripts/figures.py
    tables/                LaTeX tables emitted by scripts/tables.py
```

A `run_id` has the form `{short_name}_{lang}_{n}_{seed}`,
e.g. `xlmr_lug_250_42`.

## Reproducing the results

The grid is fully reproducible given the configurations and seeds in
`configs/base.yaml`. It comprises 2 models x 4 languages x 5 sizes x 5 seeds,
with a single seed at the `full` size (training data is deterministic at that
scale), giving 168 runs.

### Colab

Open `colab/finetune_colab.ipynb` and run the cells top to bottom. The
notebook clones this repository, installs dependencies, mounts Google Drive
to persist results across sessions, and invokes `scripts/run_grid.sh`. The
runner is idempotent: completed runs (those with a JSON log in
`results/logs/`) are skipped on re-execution.

### Local

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.check_class_coverage
bash scripts/smoke_test.sh
bash scripts/run_grid.sh
python -m scripts.aggregate
python -m scripts.figures
python -m scripts.tables
```

A CUDA GPU is required for training in a reasonable time. Apple Silicon
(MPS) works but is impractical for the full grid.

### Slicing the grid

Slice through environment variables before invoking `run_grid.sh`:

```
MODELS=xlm-r LANGS=lug NS="100 250" SEEDS="42 123" bash scripts/run_grid.sh
```

## Output schema

`results/predictions/{run_id}.parquet`:

| column            | type    | description                                    |
| ----------------- | ------- | ---------------------------------------------- |
| `id`              | int64   | row index into the MasakhaNEWS test split      |
| `text`            | string  | input text (raw `text` column)                 |
| `true_label`      | int64   | gold label index                               |
| `pred_label`      | int64   | argmax of softmax over the seven classes       |
| `logit_{label}`   | float32 | unnormalised score per class                   |
| `p_{label}`       | float32 | softmax probability per class                  |

`results/metrics.csv` carries one row per run with weighted F1 on validation
and test, run identifiers, and wall time. `results/logs/{run_id}.json`
records the full configuration, training time, and class counts of the
training subsample.

## Configuration

`configs/base.yaml` defines the shared experimental setup:

| field                     | value                                  |
| ------------------------- | -------------------------------------- |
| labels                    | seven MasakhaNEWS categories           |
| languages                 | lug, run, sna, swa                     |
| data_scales               | 100, 250, 500, 750, full               |
| seeds                     | 42, 123, 456, 789, 1024                |
| text_field                | text                                   |
| optimizer                 | AdamW                                  |
| learning_rate             | 2.0e-5                                 |
| weight_decay              | 0.01                                   |
| warmup_ratio              | 0.10                                   |
| max_epochs                | 20                                     |
| early_stopping_patience   | 10 (on weighted validation F1)         |
| max_length                | 256                                    |

Per-scale training batch sizes are set in `batch_size_schedule`. Evaluation
batch size is fixed at 64.

## Reporting

After at least some runs have completed, generate the visual and tabular
artefacts:

```
python -m scripts.figures
python -m scripts.tables
```

Figures are written to `results/figures/` as both PDF (vector, paper) and
PNG (raster). The set comprises:

- `f1_vs_n.{pdf,png}` - test weighted F1 against training-set size,
  one panel per language with both models overlaid and seed-level error bars.
- `class_distribution.{pdf,png}` - stacked bars of class counts per language
  for train, validation, and test.
- `per_class_f1_{model}_{n}.{pdf,png}` - heatmap of per-class F1 by language.
- `confusion_{model}_{lang}_{n}.{pdf,png}` - confusion matrix per
  (model, language, scale).
- `training_time_vs_n.{pdf,png}` - mean wall-clock training time by scale.
- `train_history_{run_id}.{pdf,png}` - validation F1 and loss against epoch
  for sampled runs.

LaTeX tables are written to `results/tables/`:

- `main_results.tex` - mean F1 by (model, language, scale).
- `per_class_f1_{n}.tex` - per-class F1 at one scale.
- `dataset_stats.tex` - split sizes by language.

Both scripts are idempotent and tolerate missing runs - they generate from
whatever predictions and metrics are present.

## Compute

A single trial requires roughly 5 to 20 minutes on an NVIDIA T4 depending on
training-set size and language. The full grid is approximately 40 to 50
GPU-hours.

## Dependencies

Python 3.11 with the packages pinned in `requirements.txt`.
