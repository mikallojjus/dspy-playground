"""
Shared utilities for training scripts.

Provides:
- Timestamp-based model filename generation
- Comprehensive results saving/loading
- Results comparison utilities

Usage:
    from src.training.training_utils import generate_model_filename, save_training_results

    model_path, results_path = generate_model_filename("claim_extractor")
    optimized.save(str(model_path))

    results = {
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        ...
    }
    save_training_results(results_path, results)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def generate_model_filename(model_type: str) -> tuple[Path, Path]:
    """
    Generate timestamp-based filenames for model and results in a dedicated folder.

    Each training run gets its own folder containing both model and results files.

    Args:
        model_type: Type of model (e.g., "claim_extractor", "entailment_validator")

    Returns:
        Tuple of (model_path, results_path)

    Example:
        >>> model_path, results_path = generate_model_filename("claim_extractor")
        >>> print(model_path)
        models/claim_extractor_20250110_143022/model.json
        >>> print(results_path)
        models/claim_extractor_20250110_143022/results.json
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a dedicated folder for this training run
    run_folder = Path("models") / f"{model_type}_{timestamp}"

    # Model and results files in the same folder
    model_path = run_folder / "model.json"
    results_path = run_folder / "results.json"

    # Ensure directory exists
    run_folder.mkdir(parents=True, exist_ok=True)

    return model_path, results_path


def save_training_results(results_path: Path, results: dict) -> None:
    """
    Save training results to JSON file.

    Args:
        results_path: Path to save results JSON
        results: Dictionary containing training results

    Example:
        >>> results = {
        ...     "model_path": "models/claim_extractor_20250110_143022.json",
        ...     "timestamp": "2025-01-10T14:30:22",
        ...     "baseline": {"score": 0.65},
        ...     "optimized": {"score": 0.89},
        ... }
        >>> save_training_results(Path("results/results.json"), results)
    """
    # Ensure directory exists
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with pretty formatting
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_all_results(
    results_dir: Path = None, model_type: Optional[str] = None
) -> list[dict]:
    """
    Load all training results files, optionally filtered by model type.

    Searches both new folder structure (models/*/results.json) and old flat structure
    (results/*_results.json) for backward compatibility.

    Args:
        results_dir: Directory containing model folders (default: models/)
        model_type: Optional filter by model type (e.g., "claim_extractor")

    Returns:
        List of result dictionaries, sorted by timestamp (newest first)

    Example:
        >>> results = load_all_results(model_type="claim_extractor")
        >>> for r in results[:3]:
        ...     print(f"{r['timestamp']}: {r['optimized']['score']:.3f}")
        2025-01-10T14:30:22: 0.890
        2025-01-09T10:15:30: 0.850
        2025-01-08T16:45:12: 0.820
    """
    if results_dir is None:
        results_dir = Path("models")

    results = []

    # Search new folder structure: models/*/results.json
    if results_dir.exists():
        # Find all subdirectories that match the model_type pattern
        if model_type:
            pattern = f"{model_type}_*"
        else:
            pattern = "*"

        for folder in results_dir.glob(pattern):
            if folder.is_dir():
                results_file = folder / "results.json"
                if results_file.exists():
                    try:
                        with open(results_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            # Add folder name for reference
                            data["results_folder"] = folder.name
                            data["results_file"] = str(results_file.relative_to(results_dir))
                            results.append(data)
                    except Exception as e:
                        print(f"Warning: Failed to load {results_file}: {e}")

    # Also check old flat structure for backward compatibility: results/*_results.json
    old_results_dir = Path("results")
    if old_results_dir.exists():
        pattern = f"{model_type}_*_results.json" if model_type else "*_results.json"
        result_files = list(old_results_dir.glob(pattern))

        for file_path in result_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Add filename for reference
                    data["results_file"] = file_path.name
                    data["results_folder"] = None  # Indicate old format
                    results.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

    # Sort by timestamp (newest first)
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return results


def format_metric_comparison(baseline: float, optimized: float) -> str:
    """
    Format metric comparison as "baseline → optimized (+delta)".

    Args:
        baseline: Baseline metric value
        optimized: Optimized metric value

    Returns:
        Formatted string

    Example:
        >>> format_metric_comparison(0.65, 0.89)
        '0.650 → 0.890 (+0.240)'
    """
    delta = optimized - baseline
    sign = "+" if delta >= 0 else ""
    return f"{baseline:.3f} → {optimized:.3f} ({sign}{delta:.3f})"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2m 30s", "1h 15m")

    Example:
        >>> format_duration(150)
        '2m 30s'
        >>> format_duration(3665)
        '1h 1m'
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
