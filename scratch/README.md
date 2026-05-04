# From-scratch transformer

Trains a small Transformer encoder (4 layers, 4 heads, hidden size 256,
feedforward 512, approximately 5M parameters) from random initialisation on
the [MasakhaNEWS](https://huggingface.co/datasets/masakhane/masakhanews)
classification benchmark for four agglutinative Bantu languages (Luganda,
Rundi, chiShona, Kiswahili) at controlled training-set sizes. No pretraining
is used; tokenisation is performed by a per-language SentencePiece unigram
model trained on the target-language training split.

The companion fine-tuning track lives at
[SACA-Lab/Finetuning](https://github.com/SACA-Lab/Finetuning) and the n-gram
baselines at [SACA-Lab/Ngram](https://github.com/SACA-Lab/Ngram). All three
share the same dataset preprocessing, label set, seeds, and held-out splits.

## Repository layout

```
configs/
    base.yaml              shared training configuration
src/
    data.py                MasakhaNEWS loader and stratified subsampler
    tokenizer.py           SentencePiece unigram tokeniser train/load
    model.py               encoder-only Transformer classifier
    train.py               single-trial training entry point
    predict.py             test-set inference and parquet export
    metrics.py             weighted F1 and paired t-test
scripts/
    train_tokenizer.py     train one tokeniser per language (run once)
    smoke_test.sh          single end-to-end run for pipeline validation
    check_class_coverage.py per-split and per-subsample class counts
    run_grid.sh            full grid runner; resumable
    aggregate.py           text summary tables from results/metrics.csv
    figures.py             publication figures (PDF and PNG)
    tables.py              LaTeX tables
colab/
    train_colab.ipynb      Colab notebook
    build_notebook.py      regenerates the notebook
results/
    tokenizers/{lang}.json   per-language SentencePiece tokeniser
    predictions/{run_id}.parquet
    logs/{run_id}.json
    checkpoints/             cleared after each run
    metrics.csv              appended per run
    figures/                 figures emitted by scripts/figures.py
    tables/                  LaTeX tables emitted by scripts/tables.py
```

A `run_id` has the form `{short_name}_{lang}_{n}_{seed}`, e.g.
`scratch_lug_250_42`.

## Reproducing the results

The grid is fully reproducible given the configuration and seeds in
`configs/base.yaml`. It comprises 4 languages x 5 sizes x 5 seeds with a
single seed at the `full` size, giving 84 runs.

### Local

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.train_tokenizer
python -m scripts.check_class_coverage
bash scripts/smoke_test.sh
bash scripts/run_grid.sh
python -m scripts.aggregate
python -m scripts.figures
python -m scripts.tables
```

A CUDA GPU is required for training in a reasonable time.

### Colab

Open `colab/train_colab.ipynb` and run the cells top to bottom. The notebook
clones this repository, installs dependencies, mounts Google Drive to persist
results across sessions, trains the per-language tokenisers, and invokes the
grid runner.

### Slicing the grid

```
LANGS=lug NS="100 250 500" SEEDS="42 123" bash scripts/run_grid.sh
```

The runner is idempotent: any run with an existing log is skipped.

## Output schema

`results/predictions/{run_id}.parquet` carries one row per test instance:

| column            | type    | description                              |
| ----------------- | ------- | ---------------------------------------- |
| `id`              | int64   | row index into the MasakhaNEWS test split |
| `text`            | string  | input text                                |
| `true_label`      | int64   | gold label index                          |
| `pred_label`      | int64   | argmax of softmax over the seven classes  |
| `logit_{label}`   | float32 | unnormalised score per class              |
| `p_{label}`       | float32 | softmax probability per class             |

`results/metrics.csv` carries one row per run with weighted F1 on the test
and validation splits, run identifiers, and wall time.
`results/logs/{run_id}.json` stores the configuration, parameter count,
training time, and the per-epoch trainer log history.

## Configuration

| field                     | value                                  |
| ------------------------- | -------------------------------------- |
| labels                    | seven MasakhaNEWS categories           |
| languages                 | lug, run, sna, swa                     |
| data_scales               | 100, 250, 500, 750, full               |
| seeds                     | 42, 123, 456, 789, 1024                |
| text_field                | text                                   |
| vocab_size                | 8000 (SentencePiece unigram)           |
| d_model                   | 256                                    |
| n_layers                  | 4                                      |
| n_heads                   | 4                                      |
| d_ff                      | 512                                    |
| dropout                   | 0.1                                    |
| optimizer                 | AdamW                                  |
| learning_rate             | 5.0e-4                                 |
| weight_decay              | 0.01                                   |
| warmup_ratio              | 0.10                                   |
| max_epochs                | 50                                     |
| early_stopping_patience   | 10 (on validation loss)                |
| max_length                | 256                                    |

## Reporting

After at least some runs have completed:

```
python -m scripts.figures
python -m scripts.tables
```

Figures are written to `results/figures/` as PDF and PNG. Tables are written
to `results/tables/` as standalone `.tex` files.

## Compute

Each trial requires roughly 3 to 15 minutes on an NVIDIA T4 depending on
training-set size. Tokeniser training is CPU-only and takes approximately one
minute per language. The full grid is approximately 7 to 15 GPU-hours.

## Dependencies

Python 3.11 with the packages pinned in `requirements.txt`.
