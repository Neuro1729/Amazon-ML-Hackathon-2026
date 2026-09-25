"""
Centralized, Dynamic Configuration & Hardware Discovery Module.

Paths are always relative to the cloned repository root (parent of ``src/``).
Works the same after ``git clone`` on Colab, local machines, or Kaggle.
Optional env overrides: ``DATASET_DIR``, ``OUTPUT_DIR``, ``RESULTS_DIR``, ``LOGS_DIR``.
"""

import os
import sys
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


def resolve_project_root(start: Optional[Path] = None) -> Path:
    """
    Resolve the repository root that contains ``src/config.py``.

    Prefers the package location (``__file__``), then walks upward from
    ``start`` / cwd so notebooks started in a subfolder still find the clone.
    """
    if start is None:
        start = Path.cwd()
    start = Path(start).resolve()

    package_root = Path(__file__).resolve().parent.parent
    if (package_root / "src" / "config.py").exists():
        return package_root

    for candidate in [start, *start.parents]:
        if (candidate / "src" / "config.py").exists():
            return candidate
    return package_root


def is_colab() -> bool:
    """True when running inside Google Colab."""
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return True
    if os.environ.get("GOOGLE_COLAB", "").lower() in {"1", "true"}:
        return True
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def is_kaggle() -> bool:
    """True when running inside a Kaggle notebook/kernel."""
    return Path("/kaggle/input").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


# Base Paths: always the cloned repo root
PROJECT_ROOT = resolve_project_root()


def get_hardware_info() -> Dict[str, Any]:
    """Dynamically queries CPU, RAM, and CUDA GPU specs."""
    cuda_avail = HAS_TORCH and torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_avail else 0
    cuda_ver = torch.version.cuda if cuda_avail else "N/A"
    torch_ver = torch.__version__ if HAS_TORCH else "N/A"
    
    gpus = []
    if cuda_avail:
        for i in range(gpu_count):
            gpus.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
            })
            
    vm = psutil.virtual_memory()
    info = {
        "pytorch_version": torch_ver,
        "cuda_available": cuda_avail,
        "cuda_version": cuda_ver,
        "gpu_count": gpu_count,
        "gpus": gpus,
        "cpu_count": os.cpu_count() or 1,
        "total_ram_gb": round(vm.total / (1024**3), 2),
        "available_ram_gb": round(vm.available / (1024**3), 2),
    }
    return info


def print_gpu_info() -> None:
    """
    Prints hardware and GPU information (Colab / Kaggle / local).
    """
    info = get_hardware_info()
    print(f"Runtime: Colab={is_colab()} Kaggle={is_kaggle()}")
    print(f"GPU count: {info['gpu_count']}")
    if info["gpu_count"] > 0:
        for i, g in enumerate(info["gpus"]):
            print(f"GPU {i}: {g['name']} ({g['total_memory_gb']} GB)")
    else:
        print("GPU 0: None detected (CPU fallback enabled)")
    print(f"CUDA version: {info['cuda_version']}")
    print(f"PyTorch CUDA availability: {info['cuda_available']}")
    print(f"CPU count: {info['cpu_count']}")
    print(f"RAM available: {info['available_ram_gb']} / {info['total_ram_gb']} GB")
    print(f"Project root: {PROJECT_ROOT}")
    out = globals().get("OUTPUT_DIR", PROJECT_ROOT / "output")
    data = globals().get("DATASET_DIR", PROJECT_ROOT / "dataset")
    print(f"Output dir: {out}")
    print(f"Dataset dir: {data}")


def ensure_dataset_ready() -> Path:
    """Raise a clear error if train/test TSVs are missing (common Colab setup miss)."""
    required = [
        TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GROUND_TRUTH_PATH,
        TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH,
    ]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset files missing. Put the competition `dataset/train` and "
            "`dataset/test` folders under this clone, or set DATASET_DIR, then "
            f"call refresh_paths(). Missing:\n  - " + "\n  - ".join(missing)
        )
    return DATASET_DIR


def release_memory() -> None:
    """Explicitly releases CPU and GPU memory between major pipeline stages."""
    import gc
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


class StageTimer:
    """Tracks elapsed time across major pipeline stages for transparent monitoring."""
    def __init__(self):
        import time
        self.timings: Dict[str, float] = {}
        self._starts: Dict[str, float] = {}
        self.total_start: float = time.time()

    def start(self, stage: str) -> None:
        import time
        self._starts[stage] = time.time()

    def stop(self, stage: str) -> float:
        import time
        if stage in self._starts:
            dur = time.time() - self._starts[stage]
            self.timings[stage] = dur
            return dur
        return 0.0

    def print_summary(self) -> None:
        import time
        total_runtime = time.time() - self.total_start
        print("\nStage Timing Summary:")
        print(f"Data loading time:          {self.timings.get('data_loading', 0.0):.2f}s")
        print(f"Normalization time:         {self.timings.get('normalization', 0.0):.2f}s")
        print(f"Candidate-generation time:  {self.timings.get('candidate_generation', 0.0):.2f}s")
        print(f"Feature-extraction time:    {self.timings.get('feature_extraction', 0.0):.2f}s")
        print(f"Training time:              {self.timings.get('training', 0.0):.2f}s")
        print(f"Inference time:             {self.timings.get('inference', 0.0):.2f}s")
        print(f"Total runtime:              {total_runtime:.2f}s")



def _looks_like_dataset_dir(path: Path) -> bool:
    """True if path contains train/ and test/ (competition layout)."""
    return path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()


def _candidate_dataset_dirs(base_hint: Optional[Path] = None) -> List[Path]:
    """Ordered list of relative / common places to look for the dataset."""
    root = PROJECT_ROOT
    cwd = Path.cwd().resolve()
    candidates: List[Path] = []

    env_dir = os.environ.get("DATASET_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser().resolve())

    if base_hint:
        hint = Path(base_hint).expanduser().resolve()
        candidates.extend([hint, hint / "dataset", hint / "student_resource" / "dataset"])

    # Prefer paths next to the clone (Colab / local / sibling student_resource)
    candidates.extend([
        root / "dataset",
        root / "student_resource" / "dataset",
        root / "data" / "dataset",
        root.parent / "student_resource" / "dataset",
        root.parent / "dataset",
        cwd / "dataset",
        cwd / "student_resource" / "dataset",
        cwd.parent / "student_resource" / "dataset",
    ])

    # Colab common mounts
    if Path("/content").exists():
        candidates.extend([
            Path("/content/dataset"),
            Path("/content/student_resource/dataset"),
            Path("/content/drive/MyDrive/amazon_ml/student_resource/dataset"),
            Path("/content/drive/MyDrive/student_resource/dataset"),
        ])

    # Kaggle optional fallback (never preferred over relative clone paths)
    if is_kaggle():
        candidates.append(Path("/kaggle/input"))

    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _find_dataset_under(root: Path, max_depth: int = 3) -> Optional[Path]:
    """
    Find a train/+test/ dataset directory under root without deep walks.
    Depth-limited to keep Colab/Kaggle discovery fast.
    """
    if not root.exists():
        return None
    if _looks_like_dataset_dir(root):
        return root

    # Shallow BFS
    frontier = [root]
    for _ in range(max_depth):
        next_frontier: List[Path] = []
        for node in frontier:
            try:
                children = [p for p in node.iterdir() if p.is_dir()]
            except OSError:
                continue
            for child in children:
                if _looks_like_dataset_dir(child):
                    return child
                next_frontier.append(child)
        frontier = next_frontier
    return None


def discover_dataset_paths(base_hint: Optional[Path] = None) -> Dict[str, Path]:
    """
    Discover train/test TSV paths relative to the cloned repository.

    Search order: ``DATASET_DIR`` env → repo-relative folders → Colab mounts →
    sibling ``student_resource`` → optional Kaggle input.
    """
    dataset_dir: Optional[Path] = None
    for candidate in _candidate_dataset_dirs(base_hint):
        if not candidate.exists():
            continue
        if _looks_like_dataset_dir(candidate):
            dataset_dir = candidate
            break
        depth = 3 if candidate.name in {"input", "content", "MyDrive"} else 2
        found = _find_dataset_under(candidate, max_depth=depth)
        if found is not None:
            dataset_dir = found
            break

    if dataset_dir is None:
        dataset_dir = PROJECT_ROOT / "dataset"
        logger.warning(
            "[CONFIG] Dataset not found yet. Expected at %s "
            "(or set DATASET_DIR). Place train/ and test/ under it.",
            dataset_dir,
        )
    else:
        logger.info("[CONFIG] Using dataset at %s", dataset_dir)

    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    def find_file(d: Path, pattern: str) -> Path:
        if d.is_dir():
            matches = sorted(d.glob(pattern))
            if matches:
                return matches[0]
        split = "train" if d.name == "train" else "test"
        conventional = {
            "*source1*.tsv": f"{split}_source1.tsv",
            "*source2*.tsv": f"{split}_source2.tsv",
            "*source3*.tsv": f"{split}_source3.tsv",
            "*ground_truth*.tsv": "train_ground_truth.tsv",
        }
        return d / conventional.get(pattern, pattern.replace("*", ""))

    return {
        "dataset_dir": dataset_dir,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "train_s1": find_file(train_dir, "*source1*.tsv"),
        "train_s2": find_file(train_dir, "*source2*.tsv"),
        "train_s3": find_file(train_dir, "*source3*.tsv"),
        "train_gt": find_file(train_dir, "*ground_truth*.tsv"),
        "test_s1": find_file(test_dir, "*source1*.tsv"),
        "test_s2": find_file(test_dir, "*source2*.tsv"),
        "test_s3": find_file(test_dir, "*source3*.tsv"),
    }


def resolve_runtime_dirs() -> Dict[str, Path]:
    """
    Writable dirs always live under the clone (or env overrides).

    Never hardcodes ``/kaggle/working`` so Colab/local clones stay self-contained.
    """
    def _env_or_relative(env_key: str, relative: str) -> Path:
        raw = os.environ.get(env_key)
        if raw:
            return Path(raw).expanduser().resolve()
        return PROJECT_ROOT / relative

    return {
        "output": _env_or_relative("OUTPUT_DIR", "output"),
        "results": _env_or_relative("RESULTS_DIR", "results"),
        "logs": _env_or_relative("LOGS_DIR", "logs"),
    }


def apply_discovered_paths(paths: Optional[Dict[str, Path]] = None) -> Dict[str, Path]:
    """Bind module-level path globals from a discovery result."""
    global DISCOVERED_PATHS, DATASET_DIR, TRAIN_DIR, TEST_DIR
    global TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GROUND_TRUTH_PATH
    global TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH
    global OUTPUT_DIR, RESULTS_DIR, LOGS_DIR
    global SUBMISSION_MATCHING_PATH, SUBMISSION_CANDIDATE_PATH, SUBMISSION_PREDICTIONS_PATH

    DISCOVERED_PATHS = paths or discover_dataset_paths()
    DATASET_DIR = DISCOVERED_PATHS["dataset_dir"]
    TRAIN_DIR = DISCOVERED_PATHS["train_dir"]
    TEST_DIR = DISCOVERED_PATHS["test_dir"]
    TRAIN_S1_PATH = DISCOVERED_PATHS["train_s1"]
    TRAIN_S2_PATH = DISCOVERED_PATHS["train_s2"]
    TRAIN_S3_PATH = DISCOVERED_PATHS["train_s3"]
    TRAIN_GROUND_TRUTH_PATH = DISCOVERED_PATHS["train_gt"]
    TEST_S1_PATH = DISCOVERED_PATHS["test_s1"]
    TEST_S2_PATH = DISCOVERED_PATHS["test_s2"]
    TEST_S3_PATH = DISCOVERED_PATHS["test_s3"]

    runtime = resolve_runtime_dirs()
    OUTPUT_DIR = runtime["output"]
    RESULTS_DIR = runtime["results"]
    LOGS_DIR = runtime["logs"]
    for p in (OUTPUT_DIR, RESULTS_DIR, LOGS_DIR):
        p.mkdir(parents=True, exist_ok=True)

    SUBMISSION_MATCHING_PATH = OUTPUT_DIR / "matching_results.tsv"
    SUBMISSION_CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
    SUBMISSION_PREDICTIONS_PATH = OUTPUT_DIR / "predictions.tsv"
    return DISCOVERED_PATHS


def refresh_paths(base_hint: Optional[Path] = None) -> Dict[str, Path]:
    """
    Re-run discovery after mounting Drive / uploading data in Colab.

    Example::
        from src.config import refresh_paths, print_runtime_paths
        refresh_paths("/content/drive/MyDrive/student_resource")
        print_runtime_paths()
    """
    global PROJECT_ROOT
    PROJECT_ROOT = resolve_project_root()
    return apply_discovered_paths(discover_dataset_paths(base_hint))


def print_runtime_paths() -> None:
    """Print resolved project / dataset / output paths for debugging."""
    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Dataset dir:   {DATASET_DIR}")
    print(f"Train dir:     {TRAIN_DIR}  exists={TRAIN_DIR.exists()}")
    print(f"Test dir:      {TEST_DIR}  exists={TEST_DIR.exists()}")
    print(f"Output dir:    {OUTPUT_DIR}")
    print(f"Results dir:   {RESULTS_DIR}")
    print(f"Colab:         {is_colab()}  Kaggle: {is_kaggle()}")


# Discover paths dynamically (relative to this clone)
DISCOVERED_PATHS: Dict[str, Path] = {}
DATASET_DIR: Path
TRAIN_DIR: Path
TEST_DIR: Path
TRAIN_S1_PATH: Path
TRAIN_S2_PATH: Path
TRAIN_S3_PATH: Path
TRAIN_GROUND_TRUTH_PATH: Path
TEST_S1_PATH: Path
TEST_S2_PATH: Path
TEST_S3_PATH: Path
OUTPUT_DIR: Path
RESULTS_DIR: Path
LOGS_DIR: Path
SUBMISSION_MATCHING_PATH: Path
SUBMISSION_CANDIDATE_PATH: Path
SUBMISSION_PREDICTIONS_PATH: Path

apply_discovered_paths()

# Reproducibility Seed
RANDOM_SEED = 42

# Candidate Generation Hyperparameters
DEFAULT_K_NAME = 25
DEFAULT_K_ADDRESS = 20
DEFAULT_K_COMBINED = 25
DEFAULT_K_CHAR_TFIDF = 25
BM25_K_VALUES = [1, 5, 10, 20, 50]

# Text Retrieval Settings
CHAR_NGRAM_RANGE = (3, 5)
CHAR_TFIDF_MAX_FEATURES = 150000

# Memory-Safe Chunk Sizes
def get_device() -> Any:
    """Returns primary torch device (cuda:0 if available, else cpu)."""
    if HAS_TORCH and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu") if HAS_TORCH else "cpu"


def get_available_devices() -> List[Any]:
    """
    Returns list of all available PyTorch devices without assuming 2 GPUs.
    If 2 GPUs present: [torch.device("cuda:0"), torch.device("cuda:1")].
    If 1 GPU present: [torch.device("cuda:0")].
    If 0 GPUs: [torch.device("cpu")].
    """
    if HAS_TORCH and torch.cuda.is_available():
        count = torch.cuda.device_count()
        return [torch.device(f"cuda:{i}") for i in range(count)]
    return [torch.device("cpu")] if HAS_TORCH else ["cpu"]


def get_optimal_batch_and_chunk_sizes() -> Tuple[int, int]:
    """
    Dynamically adjusts streaming chunk size and retrieval batch size based on:
    - Available CPU RAM
    - Number of available CUDA GPUs
    - User environment overrides
    """
    # Check environment variable overrides first
    env_chunk = os.environ.get("CHUNK_SIZE")
    env_batch = os.environ.get("RETRIEVAL_BATCH")
    if env_chunk and env_batch:
        return int(env_chunk), int(env_batch)

    hw = get_hardware_info()
    avail_ram = hw["available_ram_gb"]
    n_gpus = hw["gpu_count"]

    if n_gpus >= 2 and avail_ram >= 12.0:
        chunk_size = 50000
        batch_size = 10000
    elif n_gpus == 1 and avail_ram >= 10.0:
        chunk_size = 40000
        batch_size = 8000
    elif avail_ram >= 10.0:
        chunk_size = 30000
        batch_size = 5000
    else:
        chunk_size = 20000
        batch_size = 4000

    if env_chunk:
        chunk_size = int(env_chunk)
    if env_batch:
        batch_size = int(env_batch)

    return chunk_size, batch_size


# Memory-Safe Chunk Sizes (Dynamically Determined)
DEFAULT_CHUNK_SIZE, DEFAULT_RETRIEVAL_BATCH = get_optimal_batch_and_chunk_sizes()

# Precision-Oriented Ranking Model Parameters (LightGBM)
MODEL_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 45,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# Target Evaluation Metric: Macro F0.5
BETA = 0.5

# Configurable Data Sampling Parameters (Zero Hardcoding)
# Allows seamless scaling between quick notebook verification and full multi-million row inference
SAMPLE_S1_ROWS: Optional[int] = int(os.environ.get("SAMPLE_S1_ROWS", 25000)) if os.environ.get("SAMPLE_S1_ROWS", "25000") != "0" else None
SAMPLE_QUERY_ROWS: Optional[int] = int(os.environ.get("SAMPLE_QUERY_ROWS", 20000)) if os.environ.get("SAMPLE_QUERY_ROWS", "20000") != "0" else None
SAMPLE_ACTIVE_QUERIES: Optional[int] = int(os.environ.get("SAMPLE_ACTIVE_QUERIES", 10000)) if os.environ.get("SAMPLE_ACTIVE_QUERIES", "10000") != "0" else None
MAX_TEST_QUERIES: Optional[int] = int(os.environ.get("MAX_TEST_QUERIES", 10000)) if os.environ.get("MAX_TEST_QUERIES", "10000") != "0" else None


def save_threshold_config(
    abs_threshold: float,
    margin_threshold: float,
    path: Optional[Path] = None,
    extra_metrics: Optional[Dict[str, Any]] = None
) -> Path:
    """Persists optimal validation thresholds to a JSON artifact for reproducible test inference."""
    import json
    save_path = path or (RESULTS_DIR / "threshold_config.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "abs_threshold": float(abs_threshold),
        "margin_threshold": float(margin_threshold),
        "beta": float(BETA),
        "metrics": extra_metrics or {}
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"[CONFIG] Saved threshold config to {save_path}")
    return save_path


def load_threshold_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads optimal validation thresholds from disk, with fallback defaults if missing."""
    import json
    load_path = path or (RESULTS_DIR / "threshold_config.json")
    if load_path.exists():
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"[CONFIG] Loaded frozen threshold config from {load_path}")
            return data
        except Exception as e:
            logger.warning(f"[CONFIG] Failed reading {load_path}: {e}. Using defaults.")
    # Safe defaults
    return {
        "abs_threshold": 0.65,
        "margin_threshold": 0.05,
        "beta": BETA,
        "metrics": {}
    }

