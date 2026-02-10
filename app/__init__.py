from flask import Flask

from app.routes import register_routes


class DefaultConfig:
    SECRET_KEY = "dev-secret"
    BASE_MODEL_NAME = "distilgpt2"
    PROMPTS_FILE = "prompts/personal_assistant_prompts.md"
    DEFAULT_MAX_NEW_TOKENS = 120


def create_app(config_object: object | None = None) -> Flask:
    app = Flask(__name__, template_folder="../templates")

    app.config.from_object(DefaultConfig)
    if config_object is not None:
        app.config.from_object(config_object)

    register_routes(app)
    return app
