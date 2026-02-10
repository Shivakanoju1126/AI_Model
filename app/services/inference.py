from __future__ import annotations

from dataclasses import dataclass

from app.services.prompt_loader import PromptLibrary


@dataclass
class InferenceConfig:
    model_name: str
    prompts_file: str
    max_new_tokens: int


class PersonalAssistant:
    """Inference service for a LoRA-capable personal assistant."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.prompt_library = PromptLibrary(config.prompts_file)
        self._generator = None

    @property
    def generator(self):
        if self._generator is None:
            from transformers import pipeline

            self._generator = pipeline(
                task="text-generation",
                model=self.config.model_name,
            )
        return self._generator

    def build_prompt(self, user_message: str) -> str:
        persona = self.prompt_library.get("persona", "You are a helpful personal assistant.")
        style = self.prompt_library.get("response style", "Be concise and practical.")
        return (
            f"System Persona:\n{persona}\n\n"
            f"Style Rules:\n{style}\n\n"
            f"User Request:\n{user_message}\n\n"
            "Assistant Response:"
        )

    def respond(self, user_message: str) -> str:
        prompt = self.build_prompt(user_message)
        output = self.generator(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        generated = output[0]["generated_text"]
        return generated.replace(prompt, "").strip()
