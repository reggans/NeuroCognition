import openai

import os, time, re

import json
from typing import List, Dict


def validate_message_turns(messages: List[Dict], save_error: bool = True) -> bool:
    """
    Validates that messages alternate properly between user and assistant/model roles.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        save_error: Whether to save invalid messages to error.json

    Returns:
        bool: True if messages alternate properly, False otherwise
    """
    if not messages:
        return True

    # Skip system message if present
    start_idx = 0
    if messages[0]["role"] == "system":
        start_idx = 1

    for i in range(start_idx, len(messages) - 1):
        current_role = messages[i]["role"]
        next_role = messages[i + 1]["role"]

        # Check if same role appears consecutively
        if current_role == next_role:
            if save_error:
                error_info = {
                    "error": "Non-alternating message turns detected",
                    "position": i,
                    "messages": messages,
                }
                with open("error.json", "w") as f:
                    json.dump(error_info, f, indent=2)
            return False

        # Verify valid role pairs
        valid_pairs = {
            "user": ["assistant", "model"],
            "assistant": ["user"],
            "model": ["user"],
        }

        if next_role not in valid_pairs.get(current_role, []):
            if save_error:
                error_info = {
                    "error": f"Invalid role sequence: {current_role} -> {next_role}",
                    "position": i,
                    "messages": messages,
                }
                with open("error.json", "w") as f:
                    json.dump(error_info, f, indent=2)
            return False

    return True


class ModelWrapper:
    def __init__(
        self,
        model_name,
        model_source,
        api_key=None,
        max_new_tokens=512,
        think_budget=256,
        image_input=False,
        image_path=None,
    ):
        if image_input:
            if image_path is None:
                raise ValueError("Image path must be provided")
            if model_source not in ["vllm", "openai", "openrouter"]:
                raise NotImplementedError

        self.chat = None
        self.client = None
        self.history = None
        self.model_name = model_name
        self.model_source = model_source
        self.max_new_tokens = max_new_tokens
        self.think_budget = think_budget
        self.image_input = image_input
        self.image_path = image_path
        self.reasoning_trace = []  # Add new private attribute

        if model_source in ["openai", "openrouter", "vllm"]:
            if model_source == "vllm":
                api_key = "dummy"  # VLLM doesn't need a real API key
                base_url = f"http://{os.getenv('VLLM_URL')}:8877/v1"
            else:
                if api_key is None:
                    api_key = os.getenv("OPENAI_API_KEY") or os.getenv(
                        "OPENROUTER_API_KEY"
                    )
                    if api_key is None:
                        raise ValueError(
                            "Please set the OPENAI_API_KEY or OPENROUTER_API_KEY environment variable or pass it to the CLI."
                        )
                base_url = None
                if model_source == "openrouter":
                    base_url = "https://openrouter.ai/api/v1"

            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        else:
            raise ValueError(
                "Unsupported model source. Supported sources are: openai, openrouter, vllm."
            )

    def init_chat(self, task_prompt):
        self.history = [
            {"role": "system", "content": task_prompt},
        ]
        self.reasoning_trace = []  # Reset reasoning trace when starting new chat

    def send_message(
        self, message, max_new_tokens=None, truncate_history=False, cot=False
    ):
        if not validate_message_turns(self.history):
            raise ValueError(
                "Invalid message turn sequence detected. Check error.json for details."
            )

        # Store original response
        raw_response = None

        if self.image_input:
            self.history.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": os.path.join(self.image_path, "current.png")
                            },
                        },
                    ],
                }
            )
        else:
            self.history.append({"role": "user", "content": message})

        extra_body = {}
        if self.model_source == "vllm":
            extra_body = {
                "chat_template_kwargs": {"enable_thinking": bool(cot)},
            }
        elif self.model_source == "openrouter":
            if cot:
                extra_body = {"reasoning": {"exclude": False}}
                if "grok" not in self.model_name:
                    extra_body["reasoning"]["max_tokens"] = self.think_budget
                else:
                    extra_body["reasoning"]["enabled"] = True
            else:
                extra_body = {"reasoning": {"enabled": False}}
        try:
            raw_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                max_tokens=max_new_tokens or self.max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                extra_body=extra_body,
            )
            raw_response = raw_response.choices[0].message.content
        except:
            time.sleep(5)
            raw_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                max_tokens=max_new_tokens or self.max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                extra_body=extra_body,
            )
            raw_response = raw_response.choices[0].message.content

        # Add this code after getting raw_response but before updating history
        if cot:
            # Extract reasoning trace from response
            trace = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
            if not trace:
                trace = re.search(r"<thinking>(.*?)</thinking>", raw_response, re.DOTALL)
            if trace:
                self.reasoning_trace.append(
                    {"user_message": message, "reasoning": trace.group(1).strip()}
                )
            else:
                self.reasoning_trace.append(
                    {"user_message": message, "reasoning": raw_response.strip()}
                )

        # Parse response
        if truncate_history:
            # Remove reasoning trace and get content after </think> or </thinking>
            parsed = re.search(r"</think>(.*?)$", raw_response, re.DOTALL)
            if not parsed:
                parsed = re.search(r"</thinking>(.*?)$", raw_response, re.DOTALL)
            if parsed:
                response = parsed.group().strip()
            else:
                # If no closing tag found, limit to last 256 words
                words = raw_response.split()
                response = " ".join(words[-256:]).strip()

            # Additional truncation for qwen3 models: remove <step>, <reasoning>, <conclusion> tags and their content
            if "qwen3" in self.model_name:
                # Remove <step>...</step>, <reasoning>...</reasoning>, <conclusion>...</conclusion> (including tags)
                response = re.sub(r"<step>.*?</step>", "", response, flags=re.DOTALL)
                response = re.sub(
                    r"<reasoning>.*?</reasoning>", "", response, flags=re.DOTALL
                )
                response = re.sub(
                    r"<conclusion>.*?</conclusion>", "", response, flags=re.DOTALL
                )
                response = response.strip()

        self.history.append(
            {
                "role": "assistant",
                "content": (response if truncate_history else raw_response or response),
            }
        )

        return raw_response
