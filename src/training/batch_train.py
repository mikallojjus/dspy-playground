"""
Batch training script for overnight experimentation with different max_demos values.

Runs multiple training sessions to gather statistics for model comparison:
- Claim Extractor: 5 runs each with max_demos=4, 10, 16 (15 runs)
- Entailment Validator: 5 runs each with max_demos=4, 10, 16 (15 runs)
Total: 30 training runs

Usage:
    uv run python -m src.training.batch_train
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class TrainingConfig:
    """Represents a single training configuration."""

    def __init__(self, model_type: str, optimizer: str, max_demos: int, run_number: int):
        self.model_type = model_type
        self.optimizer = optimizer  # "BootstrapFewShot" or "MIPROv2"
        self.max_demos = max_demos
        self.run_number = run_number

    def __repr__(self):
        return f"{self.model_type} ({self.optimizer}, max_demos={self.max_demos}, run {self.run_number}/5)"


class BatchTrainer:
    """Orchestrates batch training runs."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = None
        self.total_runs = 0
        self.successful_runs = 0
        self.failed_runs = 0
        self.run_times: List[float] = []

    def setup_logging(self):
        """Setup logging to file and console."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        self.log(f"Batch Training Started at {datetime.now().isoformat()}")
        self.log("=" * 80)

    def log(self, message: str):
        """Write to both console and log file."""
        print(message)
        if self.log_file:
            self.log_file.write(message + '\n')
            self.log_file.flush()

    def generate_configs(self) -> List[TrainingConfig]:
        """Generate all training configurations."""
        configs = []

        # BootstrapFewShot optimizer (30 runs)
        # Claim extractor: 5 runs each for max_demos=4, 6, 8
        for max_demos in [4, 6, 8]:
            for run_num in range(1, 6):
                configs.append(TrainingConfig("claim_extractor", "BootstrapFewShot", max_demos, run_num))

        # Entailment validator: 5 runs each for max_demos=4, 6, 8
        for max_demos in [4, 6, 8]:
            for run_num in range(1, 6):
                configs.append(TrainingConfig("entailment_validator", "BootstrapFewShot", max_demos, run_num))

        # MIPROv2 optimizer (30 runs)
        # Claim extractor: 5 runs each for max_demos=4, 6, 8
        for max_demos in [4, 6, 8]:
            for run_num in range(1, 6):
                configs.append(TrainingConfig("claim_extractor", "MIPROv2", max_demos, run_num))

        # Entailment validator: 5 runs each for max_demos=4, 6, 8
        for max_demos in [4, 6, 8]:
            for run_num in range(1, 6):
                configs.append(TrainingConfig("entailment_validator", "MIPROv2", max_demos, run_num))

        return configs

    def validate_prerequisites(self) -> bool:
        """Check if prerequisites are met."""
        self.log("Checking prerequisites...")

        # Check datasets exist
        datasets = [
            "evaluation/claims_train.json",
            "evaluation/claims_val.json",
            "evaluation/entailment_train.json",
            "evaluation/entailment_val.json"
        ]

        for dataset in datasets:
            if not Path(dataset).exists():
                self.log(f"❌ Error: Dataset not found: {dataset}")
                return False

        self.log("✓ All datasets found")
        self.log("✓ Prerequisites validated")
        return True

    def run_training(self, config: TrainingConfig, run_index: int, total: int) -> Tuple[bool, float]:
        """Run a single training configuration."""
        start_time = time.time()

        # Progress header
        self.log("")
        self.log("=" * 80)
        self.log(f"Run {run_index}/{total}: {config}")
        self.log("=" * 80)

        # Build command based on optimizer and model type
        if config.optimizer == "BootstrapFewShot":
            if config.model_type == "claim_extractor":
                cmd = [
                    "uv", "run", "python", "-m", "src.training.train_claim_extractor",
                    "--max-demos", str(config.max_demos)
                ]
            elif config.model_type == "entailment_validator":
                cmd = [
                    "uv", "run", "python", "-m", "src.training.train_entailment_validator",
                    "--max-demos", str(config.max_demos)
                ]
            else:
                self.log(f"❌ Unknown model type: {config.model_type}")
                return False, 0
        elif config.optimizer == "MIPROv2":
            if config.model_type == "claim_extractor":
                cmd = [
                    "uv", "run", "python", "-m", "src.training.train_claim_extractor_mipro",
                    "--max-demos", str(config.max_demos)
                ]
            elif config.model_type == "entailment_validator":
                cmd = [
                    "uv", "run", "python", "-m", "src.training.train_entailment_validator_mipro",
                    "--max-demos", str(config.max_demos)
                ]
            else:
                self.log(f"❌ Unknown model type: {config.model_type}")
                return False, 0
        else:
            self.log(f"❌ Unknown optimizer: {config.optimizer}")
            return False, 0

        # Run training
        try:
            self.log(f"Command: {' '.join(cmd)}")
            self.log("")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

            # Log output
            if result.stdout:
                self.log(result.stdout)
            if result.stderr:
                self.log("STDERR:")
                self.log(result.stderr)

            duration = time.time() - start_time

            if result.returncode == 0:
                self.log("")
                self.log(f"✓ Training completed successfully in {self.format_duration(duration)}")
                self.successful_runs += 1
                self.run_times.append(duration)
                return True, duration
            else:
                self.log("")
                self.log(f"❌ Training failed with exit code {result.returncode}")
                self.failed_runs += 1
                return False, duration

        except Exception as e:
            duration = time.time() - start_time
            self.log(f"❌ Exception during training: {e}")
            self.failed_runs += 1
            return False, duration

    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
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

    def run_comparison(self, model_type: str):
        """Run comparison tool for a model type."""
        self.log("")
        self.log("=" * 80)
        self.log(f"Comparing {model_type} models...")
        self.log("=" * 80)

        cmd = [
            "uv", "run", "python", "-m", "src.cli.compare_models",
            "--model-type", model_type
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

            if result.stdout:
                self.log(result.stdout)
            if result.returncode != 0 and result.stderr:
                self.log("STDERR:")
                self.log(result.stderr)

        except Exception as e:
            self.log(f"❌ Exception during comparison: {e}")

    def run_all(self):
        """Main orchestration method."""
        try:
            self.setup_logging()

            # Validate prerequisites
            if not self.validate_prerequisites():
                self.log("❌ Prerequisites check failed. Exiting.")
                return 1

            # Generate configurations
            configs = self.generate_configs()
            self.total_runs = len(configs)

            self.log("")
            self.log(f"Generated {self.total_runs} training configurations:")
            self.log(f"  BootstrapFewShot (30 runs):")
            self.log(f"    - Claim Extractor: 15 runs (5 × max_demos 4, 10, 16)")
            self.log(f"    - Entailment Validator: 15 runs (5 × max_demos 4, 10, 16)")
            self.log(f"  MIPROv2 (30 runs):")
            self.log(f"    - Claim Extractor: 15 runs (5 × max_demos 4, 10, 16)")
            self.log(f"    - Entailment Validator: 15 runs (5 × max_demos 4, 10, 16)")
            self.log("")

            # Run all trainings
            start_time = time.time()

            for i, config in enumerate(configs, 1):
                success, duration = self.run_training(config, i, self.total_runs)

            total_duration = time.time() - start_time

            # Final summary
            self.log("")
            self.log("=" * 80)
            self.log("BATCH TRAINING COMPLETE")
            self.log("=" * 80)
            self.log("")
            self.log(f"Total runs: {self.total_runs}")
            self.log(f"Successful: {self.successful_runs}")
            self.log(f"Failed: {self.failed_runs}")
            self.log(f"Total time: {self.format_duration(total_duration)}")

            if self.run_times:
                avg_time = sum(self.run_times) / len(self.run_times)
                self.log(f"Average time per run: {self.format_duration(avg_time)}")

            # Run comparisons
            if self.successful_runs > 0:
                self.log("")
                self.run_comparison("claim_extractor")
                self.run_comparison("entailment_validator")

            self.log("")
            self.log(f"Full log saved to: {self.log_path}")

            return 0 if self.failed_runs == 0 else 1

        except KeyboardInterrupt:
            self.log("")
            self.log("=" * 80)
            self.log("❌ INTERRUPTED BY USER")
            self.log("=" * 80)
            self.log("")
            self.log(f"Completed runs: {self.successful_runs + self.failed_runs}/{self.total_runs}")
            self.log(f"Successful: {self.successful_runs}")
            self.log(f"Failed: {self.failed_runs}")
            self.log("")
            self.log(f"Partial log saved to: {self.log_path}")
            return 130  # Standard exit code for Ctrl+C

        finally:
            if self.log_file:
                self.log_file.close()


def main():
    """Entry point for batch training."""
    print("=" * 80)
    print("Batch Training Script")
    print("=" * 80)
    print()
    print("This will run 60 training sessions:")
    print("  BootstrapFewShot (30 runs):")
    print("    - Claim Extractor: 15 runs (5 runs × 3 max_demos values)")
    print("    - Entailment Validator: 15 runs (5 runs × 3 max_demos values)")
    print("  MIPROv2 (30 runs):")
    print("    - Claim Extractor: 15 runs (5 runs × 3 max_demos values)")
    print("    - Entailment Validator: 15 runs (5 runs × 3 max_demos values)")
    print()
    print()

    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"batch_training_{timestamp}.log"

    print(f"Log file: {log_path}")
    print()

    # Run batch training
    trainer = BatchTrainer(log_path)
    return trainer.run_all()


if __name__ == "__main__":
    sys.exit(main())
