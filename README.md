# Metric Match

**Metric Match** is a method that selects which text items to annotate with expensive judges (human or LLM) in order to accurately estimate inter-rater agreement metrics — such as ICC, Krippendorff's alpha, Spearman ρ, and Kendall τ — using a minimal annotation budget.

The core insight: cheap LLM judges preserve the variance structure of expensive judges. Metric Match selects a small subset of items whose inter-model agreement on cheap judges best matches the agreement on the full dataset, then uses that subset as the annotation target for expensive judges.

---

## Installation

```bash
git clone <repo-url>
cd MetricMatch
```

**Option A — Conda (recommended).** Creates a full environment with all pinned dependencies, then installs the package in editable mode:

```bash
conda env create -f environment.yml
conda activate metric_match
pip install -e .
```

**Option B — pip only.** Installs from `pyproject.toml` into your current Python environment:

```bash
pip install -e .

# With local model support (Llama, Qwen, Gemma via HuggingFace)
pip install -e ".[local-models]"
```

---

## Quick Start: Use Metric Match on Your Own Data

You need two things:
1. **Judge scores** — a CSV/DataFrame with columns `text_id`, `model_name`, `evaluation_score`
2. A **budget** — how many items you can afford to annotate

```python
import pandas as pd
from src.utils.selection_strategies import metric_matched_selection
from src.utils.match_metrics import compute_icc_pingouin, compute_krippendorff_alpha, compute_ms_components

# Load your judge scores
# Expected columns: text_id, model_name, evaluation_score
# One row per (text, model) pair
judge_scores = pd.read_csv("my_judge_scores.csv")

# Get the set of all text IDs
text_ids = judge_scores["text_id"].unique()

# Compute the full-dataset inter-model ICC (used as the matching target)
target_icc = compute_icc_pingouin(judge_scores)

# Select 50 items using Metric Match (ICC variant)
selected_ids = metric_matched_selection(
    text_ids=text_ids,
    k=50,                          # annotation budget
    im_full_df=judge_scores,       # judge scores DataFrame
    target_value=target_icc,       # metric value to match
    target_metric="icc",           # "icc", "alpha", "rho", or "tau"
    compute_ms_fn=compute_ms_components,
    compute_icc_fn=compute_icc_pingouin,
    compute_alpha_fn=compute_krippendorff_alpha,
    seed=42,
    n_candidates=20,               # number of random subsets to evaluate
)

print(f"Selected {len(selected_ids)} items for annotation.")
print(selected_ids[:10])
```

The returned `selected_ids` is an array of `text_id` values. Annotate only those items with your expensive judge to get a reliable estimate of inter-rater agreement.

---

## End-to-End Pipeline

The full pipeline runs in two steps:

### Step 1 — Get Judge Scores

Run one or more LLM judges on your dataset to produce the inter-model signal used for selection.

```bash
python -m src.experiments._1_get_judge_scores \
    --output_file data/judge_scores/my_dataset/results_my_dataset_gpt-4o-mini \
    --dataset my_dataset \
    --dimension quality \
    --judge_model openai \
    --model_name gpt-4o-mini
```

Key arguments:

| Argument | Description |
|---|---|
| `--output_file` | Output path prefix; one JSON per dimension is created |
| `--dataset` | Dataset name: `summeval`, `hanna`, `mslr`, `medval`, or your custom dataset name |
| `--dimension` | Evaluation dimension (e.g. `Coherence`, `fluency`). Omit to run all dimensions |
| `--judge_model` | API type: `openai`, `anthropic`, `gemini`, `deepseek`, `llama`, `qwen`, `gemma` |
| `--model_name` | Model identifier (e.g. `gpt-4o-mini`, `claude-3-5-sonnet`) |
| `--sample_size` | Limit to N items (default: all) |

**API configuration.** The scoring step reads API credentials from environment variables:

```bash
# OpenAI / Azure OpenAI
export OPENAI_API_BASE="https://<your-endpoint>/openai"
export OPENAI_API_KEY="<your-key>"

# Anthropic (custom endpoint)
export ANTHROPIC_API_URL="<your-anthropic-endpoint>"

# Gemini (custom endpoint)
export GEMINI_API_URL="<your-gemini-endpoint>"

# DeepSeek (custom endpoint)
export DEEPSEEK_API_URL="<your-deepseek-endpoint>"

# Llama 3.3 70B (custom endpoint)
export LLAMA33_API_URL="<your-llama-endpoint>"
```

> **Note:** The default API functions in `src/utils/api_support_functions.py` use an Azure-style endpoint format (`Ocp-Apim-Subscription-Key` header). For standard OpenAI or Anthropic APIs, you can either set the environment variables to match your provider's format, or implement your own scoring pipeline and bring pre-computed scores (see below).

**Bringing your own judge scores.** If you already have judge scores from another pipeline, format them as JSON files and place them in `data/judge_scores/<dataset>/`:

```
data/judge_scores/
  my_dataset/
    results_my_dataset_gpt-4o-mini_quality.json
    results_my_dataset_gpt-4.1_quality.json
```

Each file must follow this schema:

```json
{
  "dimension": "quality",
  "detailed_results": [
    {
      "text_id": "doc_001",
      "input_text": "The generated text being evaluated.",
      "source_text": "Optional: the source/prompt for the text.",
      "original_score": 3.5,
      "evaluation": {
        "evaluation": { "score": 4 }
      }
    }
  ]
}
```

- `text_id`: unique identifier for each text
- `original_score`: the ground-truth human score (used only for evaluation, not for selection)
- `evaluation.evaluation.score`: the model's score for this item

### Step 2 — Run Metric Match Selection Analysis

```bash
python -m src.experiments._2_variance_selection_analysis \
    --dataset my_dataset \
    --model-names gpt-4o-mini llama-3-8b \
    --data-dir data/judge_scores \
    --plots-dir results/my_run \
    --comparison-mode pairwise_average \
    --n-bootstrap 40 \
    --max-budget 50
```

To separate the cheap ensemble signal from the expensive target models:

```bash
python -m src.experiments._2_variance_selection_analysis \
    --dataset my_dataset \
    --model-names gpt-4o-mini llama-3-8b claude-3-5-sonnet gpt-4.1 \
    --ensemble-models gpt-4o-mini llama-3-8b \
    --target-models claude-3-5-sonnet gpt-4.1 \
    --data-dir data/judge_scores \
    --plots-dir results/my_run \
    --max-budget 50
```

Key arguments:

| Argument | Description |
|---|---|
| `--dataset` | Dataset name registered in `EVALUATION_AXES` |
| `--model-names` | All judge model names to load |
| `--target-models` | Models whose HM-ICC you want to estimate (default: same as `--model-names`) |
| `--ensemble-models` | Cheap models used as the variance signal (default: same as `--model-names`) |
| `--comparison-mode` | `pairwise_average` (default), `average_pairwise`, or `aggregate` |
| `--n-bootstrap` | Number of bootstrap trials per budget level (default: 40) |
| `--n-candidates` | Candidate subsets evaluated per trial (default: 20) |
| `--max-budget` | Maximum annotation budget to test (default: 50) |
| `--step-size` | Budget step size (default: 5) |
| `--plots-dir` | Directory to save result DataFrames and plots |

Results are saved as pickle DataFrames in `--plots-dir` and plots are generated automatically.

### Step 3 — Compute Win Rates and Annotations Saved

```bash
python -m src.experiments._3_win_rates --folder results/my_run
python -m src.experiments._4_annotations_saved --folder results/my_run
python -m src.experiments._5_estimation_error --folder results/my_run
```

---

## Using a Custom Dataset

Subclass `baseDataset` in `src/dataset_classes/base_dataset.py` and implement two methods:

```python
from src.dataset_classes.base_dataset import baseDataset
import pandas as pd
from typing import List

class MyDataset(baseDataset):
    def extract_data(self) -> pd.DataFrame:
        """Load your data. Must return a DataFrame with columns:
            text_id       – unique identifier for each text
            text          – the text to be evaluated
            original_score – ground-truth human score
            source_text   – (optional) context or prompt for the text
        """
        df = pd.read_csv("path/to/my_data.csv")
        return pd.DataFrame({
            "text_id": df["id"],
            "text": df["generated_text"],
            "original_score": df["human_score"],
            "source_text": df["prompt"],
        })

    def create_prompt(self, row: pd.Series) -> str:
        """Return the prompt sent to the LLM judge for this row."""
        return f"""Evaluate the quality of this text on a scale of 1-5.

Text: {row['text']}

Respond in JSON: {{"evaluation": {{"score": <1-5>}}}}"""

    def get_available_dimensions(self) -> List[str]:
        return ["quality"]
```

Then register your dataset in `_2_variance_selection_analysis.py`:

```python
EVALUATION_AXES = {
    ...
    "my_dataset": ["quality"],
}
```

And pass it to `_1_get_judge_scores.py` with `--dataset my_dataset`.

---

## Supported Datasets

| Dataset | Dimensions | Source |
|---|---|---|
| HANNA | Coherence, Complexity, Empathy, Engagement, Relevance, Surprise | [GitHub](https://github.com/dig-team/hanna-benchmark-asg) |
| MSLR | fluency, intervention, outcome, population | [AllenAI](https://github.com/allenai/mslr-annotated-dataset) |
| SummEval | coherence, consistency, fluency, relevance | [HuggingFace](https://huggingface.co/datasets/mteb/summeval) |
| MedVal | Risk | [arXiv](https://arxiv.org/pdf/2507.03152) |

Raw data files for HANNA, MSLR, and MedVal should be placed in `data/raw_data/`. SummEval is downloaded automatically from HuggingFace.

---

## Repository Structure

```
MetricMatch/
├── src/
│   ├── dataset_classes/         # Dataset loaders
│   │   ├── base_dataset.py      # Abstract base class
│   │   ├── hanna.py
│   │   ├── mslr.py
│   │   ├── medval.py
│   │   └── summ_eval.py
│   ├── experiments/             # Numbered experiment scripts
│   │   ├── _1_get_judge_scores.py      # Step 1: collect LLM judge scores
│   │   ├── _2_variance_selection_analysis.py  # Step 2: run Metric Match
│   │   ├── _3_win_rates.py             # Step 3: compute win rates
│   │   ├── _4_annotations_saved.py     # Step 4: compute annotation savings
│   │   └── _5_estimation_error.py      # Step 5: plot estimation error curves
│   └── utils/
│       ├── api_support_functions.py    # LLM API wrappers
│       ├── data_loading.py             # Load judge score JSON files
│       ├── intraclass_corr.py          # Pointwise ICC computation
│       ├── match_metrics.py            # Reliability metric functions
│       ├── plotting.py                 # Plotting utilities
│       └── selection_strategies.py     # Sampling strategies (Metric Match here)
├── data/
│   ├── judge_scores/            # JSON files output by Step 1
│   └── raw_data/                # Raw dataset files
├── scripts/                     # SLURM job scripts
├── environment.yml
└── pyproject.toml
```

