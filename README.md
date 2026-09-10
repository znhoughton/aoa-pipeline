# Reading Acquisition Order out of BabyLM Representations

### Child-Scale Training Produces Developmentally Aligned Semantic Spaces

Chenyi Ding, Zachary Nicholas Houghton, Masoud Jasbi

Code, data and results for the paper.

How much of children's vocabulary learning order can you recover from a frozen language
model's representations, and does training on developmentally plausible data help?

The pipeline takes 18 checkpoints, extracts word representations at every layer, fits a
ridge read-out to Wordbank CDI age-of-acquisition, and compares the result against the
standard psycholinguistic predictor set. Evaluation is on CDI words only: 619 words
after cleaning, all of which have at least one CHILDES context, so no corpus-AoA
training words leak into the test set.

For each checkpoint the notebook produces:

1. a ridge read-out (5-fold CV) at every layer, scored as Spearman rho against Wordbank AoA
2. semantic-structure features (cohesion / distinctiveness / entropy) layered on top of Braginsky-4
3. bootstrap CIs and pairwise comparisons against a full-data reference model
4. two robustness controls: nested layer selection, and PCA to a common 256 dimensions

## Headline result

Best-layer contextual read-out, with the *post hoc* layer choice removed by nested CV.
Full-data controls are marked †.

| Checkpoint | rho (nested) | 95% CI |
|---|---|---|
| gptbert25-mixed | 0.639 | [0.584, 0.685] |
| gptbert25-masked | 0.637 | [0.585, 0.685] |
| gpt2-babylm100m | 0.626 | [0.572, 0.674] |
| ltgbert-100m-2024 | 0.623 | [0.568, 0.677] |
| babyllama-100m | 0.618 | [0.563, 0.667] |
| gpt2-babylm10m | 0.604 | [0.545, 0.657] |
| roberta-babylm100m | 0.583 | [0.529, 0.638] |
| gptbert25-causal | 0.583 | [0.523, 0.640] |
| opt125-babylm100m | 0.577 | [0.518, 0.630] |
| opt125-babylm150m | 0.571 | [0.509, 0.627] |
| babyllama-10m | 0.571 | [0.510, 0.628] |
| opt125-full † | 0.560 | [0.500, 0.615] |
| roberta-full † | 0.525 | [0.455, 0.592] |
| elcbert-100m-2023 | 0.523 | [0.460, 0.581] |
| gptbert-100m-2024 | 0.502 | [0.438, 0.563] |
| babyllama58-2023 | 0.489 | [0.423, 0.551] |
| gpt2-full † | 0.483 | [0.415, 0.545] |
| opt125-babylm10m | 0.414 | [0.346, 0.481] |

Reference points on the same 5-fold CV: frequency alone reaches rho = 0.244 (n = 619),
and the four-predictor Braginsky baseline (frequency, concreteness, babiness, MLU)
reaches rho = 0.707 (n = 552, the words with complete norm coverage). So the best
read-outs land well above frequency but still short of the full psycholinguistic
feature set.

The PCA-256 control (`outputs/results/pca_control.csv`) reproduces the same ordering,
so differences in hidden size across models (512–768) are not driving the ranking.

## Layout

```
data/                     six input files, see data/README.md
notebooks/
  multimodel_aoa_pipeline.ipynb    the whole pipeline, Step 1 through Step 11
outputs/
  vectors/                extracted .npz representations, NOT in git (see below)
  results/                result tables (committed)
  figures/                figures (committed)
```

### Outputs

| File | Contents |
|---|---|
| `results/readout_all_models.csv` | rho for every checkpoint x representation x layer. The main table. |
| `results/readout_bootstrap_ci.csv` | best-layer contextual rho with word-level bootstrap CIs |
| `results/pairwise_vs_reference.csv` | delta-rho vs `opt125-full`, on words valid in both models |
| `results/nested_readout.csv` | nested layer selection; the paper-grade headline numbers |
| `results/nested_pairwise_vs_reference.csv` | pairwise comparisons under nested selection |
| `results/results_multimodel.csv` | semantic-structure and Braginsky-4 regressions, long format |
| `results/pca_control.csv` | main vs PCA-256 rho per checkpoint |
| `figures/fig_readout_by_model.png` | best-layer contextual read-out per model vs Braginsky-4 |
| `figures/fig_layer_profiles.png` | rho by relative depth, decoders vs encoders/hybrids |

### The vector files are not in the repo

Step 5 writes 24 `.npz` files (2 per model) totalling roughly 350 MB into
`outputs/vectors/`. These are intermediates, so they are gitignored rather than
committed. The notebook skips extraction for any model whose `.npz` files already
exist, so regenerating them means running Steps 1–5 once on a GPU. Everything
downstream reads from `outputs/vectors/`, so once the files exist the analysis steps
are cheap to rerun.

## Running it

The notebook finds the repo root by looking for `data/cdi_items_with_aoa.csv`, so it
works both from a local clone and from Colab without editing paths.

**Locally:**

```bash
git clone https://github.com/znhoughton/aoa-pipeline.git
cd aoa-pipeline
pip install -r requirements.txt
jupyter lab notebooks/multimodel_aoa_pipeline.ipynb
```

**On Colab:** put the repo in Drive as `MyDrive/aoa-pipeline` and open the notebook.
Step 2 mounts Drive and picks it up. If autodetection ever misses, set `BASE` by hand
in Step 2.

Run the cells in order. Step 1 checks the `transformers` version and installs 4.51.3 if
needed, which requires a runtime restart, then rerun from Step 1. Step 4 is a smoke test
over 10 words x 10 contexts per model, and Step 5 refuses to extract from any model that
did not pass it. If a model prints `!!` or `X`, fix it before continuing.

A GPU is needed in practice for Step 5 only. Steps 6 onward run on CPU from the cached
vectors, though the nested CV in Step 9 is the slowest of them.

## Method notes

The extraction code is tokenizer-agnostic by design, since the checkpoints span GPT-2,
OPT, Llama, RoBERTa and LTG-BERT tokenizers:

- the target word is located by **character offsets** (`return_offsets_mapping`), not by
  matching a token subsequence
- special tokens are dropped via `special_tokens_mask` rather than by assuming position 0
  is BOS (GPT-2 has none)
- layer count comes from a dummy forward pass instead of a hardcoded 1–12

Two representations are extracted per model. **Contextual** vectors average the target
word's token span over up to 50 CHILDES sentences containing it; **bare** vectors encode
the word in isolation. Words with no matched context come out NaN and are dropped
per-model, with the surviving count reported in the `n` column of the read-out tables.

The main-figure rho picks the best layer post hoc, which is optimistic. Step 9 redoes
the headline comparison with the layer chosen inside each outer training fold, and those
are the numbers in the table above.

## Models

18 checkpoints along three axes: training-data scale, architecture family, and training
objective.

| Family | Checkpoints |
|---|---|
| OPT-125M | `opt-babylm-125m-20eps-seed964` (150M), `opt-125m-strict-2023` (100M), `opt-125m-strict-small-2023` (10M), `facebook/opt-125m` † |
| GPT-2 | `babylm-baseline-100m-gpt2`, `babylm-baseline-10m-gpt2`, `openai-community/gpt2` † |
| BabyLlama | `baby-llama-58m` (2023 winner), `babyllama-100m-2024`, `babyllama-10m-2024` |
| RoBERTa | `roberta-base-strict-2023`, `FacebookAI/roberta-base` † |
| Encoder / hybrid | `ELC_BERT_baby_100M`, `gpt-bert-babylm-base`, `ltgbert-100m-2024` |
| GPT-BERT 2025 objective mix | `-mixed`, `-causal-focus`, `-masked-focus` |

† full-data control. `opt125-full` is the reference model for pairwise comparisons.
The ELC-BERT, LTG-BERT and GPT-BERT checkpoints ship custom modelling code and load with
`trust_remote_code=True`, which is why the smoke test gates them.

## Citing

If you use this code or these results, please cite the paper:

> Ding, C., Houghton, Z. N., & Jasbi, M. (2026). Reading Acquisition Order out of
> BabyLM Representations: Child-Scale Training Produces Developmentally Aligned
> Semantic Spaces. *Proceedings of the BabyLM Challenge*.

```bibtex
@inproceedings{ding-etal-2026-reading,
    title     = "Reading Acquisition Order out of {B}aby{LM} Representations: Child-Scale Training Produces Developmentally Aligned Semantic Spaces",
    author    = "Ding, Chenyi and Houghton, Zachary Nicholas and Jasbi, Masoud",
    booktitle = "Proceedings of the BabyLM Challenge",
    year      = "2026"
}
```

This entry is provisional. Replace it with the official BibTeX from the ACL Anthology
once the proceedings are published, and update `CITATION.cff` to match.

The data files in `data/` are redistributions of published resources with their own
terms; see [data/README.md](data/README.md) and cite those papers directly rather than
citing this repo for them.
