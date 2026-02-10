from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingRequest:
    model_name: str
    dataset_path: str
    output_dir: str
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16


class LoraTrainerService:
    """Thin orchestrator around the LoRA training script."""

    script_path = Path("scripts/train_lora.py")

    def start(self, payload: TrainingRequest) -> subprocess.CompletedProcess:
        command = [
            "python",
            str(self.script_path),
            "--model-name",
            payload.model_name,
            "--dataset-path",
            payload.dataset_path,
            "--output-dir",
            payload.output_dir,
            "--num-train-epochs",
            str(payload.num_train_epochs),
            "--learning-rate",
            str(payload.learning_rate),
            "--lora-r",
            str(payload.lora_r),
            "--lora-alpha",
            str(payload.lora_alpha),
        ]
        return subprocess.run(command, check=False, capture_output=True, text=True)
