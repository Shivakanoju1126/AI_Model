# LORA Personal Assistant (Flask + LoRA Training)

A maintainable Flask application that provides:

- **Chat API** (`/chat`) for interacting with a personal assistant.
- **Training API** (`/train`) to launch LoRA fine-tuning.
- **Prompt library** in Markdown (`prompts/personal_assistant_prompts.md`) for easy updates.

## Project structure

```text
app/
  __init__.py
  routes.py
  services/
    inference.py
    prompt_loader.py
    training.py
prompts/
  personal_assistant_prompts.md
scripts/
  train_lora.py
templates/
  index.html
run.py
requirements.txt
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Open `http://localhost:5000`.

## Training dataset format

Use JSONL with a `text` field. A ready-to-use sample is included at `data/sample_train.jsonl`:

```json
{"text": "User: Help me plan my day. Assistant: Sure, let's prioritize tasks..."}
{"text": "User: Summarize this email. Assistant: Here's a concise summary..."}
```

Then call `/train` with:

```json
{
  "dataset_path": "data/sample_train.jsonl",
  "output_dir": "outputs/lora-assistant",
  "model_name": "distilgpt2",
  "num_train_epochs": 1
}
```
