from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from app.services.inference import InferenceConfig, PersonalAssistant
from app.services.training import LoraTrainerService, TrainingRequest


assistant_service: PersonalAssistant | None = None
training_service = LoraTrainerService()


def _get_assistant(app: Flask) -> PersonalAssistant:
    global assistant_service
    if assistant_service is None:
        assistant_service = PersonalAssistant(
            InferenceConfig(
                model_name=app.config["BASE_MODEL_NAME"],
                prompts_file=app.config["PROMPTS_FILE"],
                max_new_tokens=app.config["DEFAULT_MAX_NEW_TOKENS"],
            )
        )
    return assistant_service


def register_routes(app: Flask) -> None:
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/chat")
    def chat():
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "message is required"}), 400

        assistant = _get_assistant(app)
        try:
            response = assistant.respond(user_message)
            return jsonify({"response": response})
        except Exception as exc:
            return jsonify({"error": f"inference unavailable: {exc}"}), 503

    @app.post("/train")
    def train():
        data = request.get_json(silent=True) or {}
        required_fields = ["dataset_path", "output_dir"]
        missing = [field for field in required_fields if not data.get(field)]
        if missing:
            return jsonify({"error": f"missing required fields: {', '.join(missing)}"}), 400

        payload = TrainingRequest(
            model_name=data.get("model_name", app.config["BASE_MODEL_NAME"]),
            dataset_path=data["dataset_path"],
            output_dir=data["output_dir"],
            num_train_epochs=int(data.get("num_train_epochs", 1)),
            learning_rate=float(data.get("learning_rate", 2e-4)),
            lora_r=int(data.get("lora_r", 8)),
            lora_alpha=int(data.get("lora_alpha", 16)),
        )

        result = training_service.start(payload)
        status_code = 200 if result.returncode == 0 else 500

        return (
            jsonify(
                {
                    "status": "success" if result.returncode == 0 else "failed",
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            ),
            status_code,
        )
