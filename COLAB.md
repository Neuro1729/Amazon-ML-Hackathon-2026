# Colab (GPU) Quick Start

All paths are relative to this repository after `git clone`. Output, results, and logs are created under the clone root (not `/kaggle/working`).

## 1. Runtime

Runtime → Change runtime type → **GPU** (T4 / L4 / A100).

## 2. Clone and install

```python
!git clone https://github.com/Neuro1729/Amazon-ML-Hackathon-2026.git
%cd Amazon-ML-Hackathon-2026
!pip install -q -r requirements.txt
```

## 3. Attach the dataset

Upload or copy `student_resource/dataset/` (with `train/` and `test/`) into one of:

- `Amazon-ML-Hackathon-2026/dataset/`
- `Amazon-ML-Hackathon-2026/student_resource/dataset/`
- Drive path, then point the env var:

```python
from google.colab import drive
drive.mount("/content/drive")

import os
os.environ["DATASET_DIR"] = "/content/drive/MyDrive/student_resource/dataset"

from src.config import refresh_paths, print_runtime_paths, print_gpu_info
refresh_paths()
print_gpu_info()
print_runtime_paths()
```

## 4. Run the pipeline

```python
!python scripts/profile_dataset.py
!python scripts/build_candidates.py --sample-s1 25000 --sample-queries 10000
!python scripts/train_model.py --sample-s1 25000 --sample-queries 10000 --max-negatives 8
!python scripts/evaluate.py
!python scripts/generate_submission.py
```

Or open `amazon-ml-hackathon.ipynb` from the clone and run all cells.

Submission files land in:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

## Notes

- PyTorch will use the Colab GPU when available. Blocking stays on sparse CPU (by design). LightGBM uses CPU by default; set `USE_LGBM_GPU=1` only if your LightGBM build has GPU support.
- Full-data runs need more RAM/time. Keep the sample flags for a first pass, then set `SAMPLE_S1_ROWS=0` (and related env vars) for a full run.
