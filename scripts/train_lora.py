from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for the personal assistant")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-path", required=True, help="JSONL file with a 'text' field")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["c_attn", "q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files=str(dataset_path))["train"]

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
