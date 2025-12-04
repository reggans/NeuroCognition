import openai

import os, time, re
import base64

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


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string with proper data URL format.

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded image as data URL (data:image/png;base64,...)
    """
    try:
        with open(image_path, "rb") as image_file:
            # Read the image file
            image_data = image_file.read()
            # Encode to base64
            base64_string = base64.b64encode(image_data).decode("utf-8")
            # Determine the image format from file extension
            file_extension = os.path.splitext(image_path)[1].lower()
            if file_extension == ".png":
                mime_type = "image/png"
            elif file_extension in [".jpg", ".jpeg"]:
                mime_type = "image/jpeg"
            elif file_extension == ".gif":
                mime_type = "image/gif"
            elif file_extension == ".webp":
                mime_type = "image/webp"
            else:
                # Default to png if unknown
                mime_type = "image/png"

            # Return as data URL
            return f"data:{mime_type};base64,{base64_string}"
    except Exception as e:
        raise ValueError(f"Failed to encode image {image_path}: {str(e)}")


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
        n_retry: int = 2,
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
        # Max number of retry attempts for a failed API call (total attempts = n_retry + 1)
        self.n_retry = max(0, n_retry)

        if model_source in ["openai", "openrouter", "vllm"]:
            if model_source == "vllm":
                api_key = "dummy"  # VLLM doesn't need a real API key
                base_url = f"http://{os.getenv('VLLM_URL')}/v1"
            elif model_source == "openai":
                if api_key is None:
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key is None:
                        raise ValueError(
                            "Please set the OPENAI_API_KEY environment variable for OpenAI or pass it to the CLI."
                        )
                base_url = None
            else:
                if api_key is None:
                    api_key = os.getenv("OPENROUTER_API_KEY")
                    if api_key is None:
                        raise ValueError(
                            "Please set the OPENROUTER_API_KEY environment variable for OpenRouter or pass it to the CLI."
                        )
                base_url = "https://openrouter.ai/api/v1"
                # OpenRouter recommends sending HTTP-Referer and X-Title headers, especially for free-tier models
                self._openrouter_headers = {}
                ref = os.getenv("OPENROUTER_REF", "")
                title = os.getenv("OPENROUTER_TITLE", "CognitiveEval")
                if ref:
                    self._openrouter_headers["HTTP-Referer"] = ref
                if title:
                    self._openrouter_headers["X-Title"] = title

            # Build OpenAI client with optional default headers and a slightly larger timeout for OpenRouter
            client_kwargs = {
                "api_key": api_key,
                "base_url": base_url,
            }
            # Attach default headers if using OpenRouter
            if model_source == "openrouter":
                if getattr(self, "_openrouter_headers", None):
                    client_kwargs["default_headers"] = self._openrouter_headers  # type: ignore
                # Increase timeout to accommodate longer generations
                client_kwargs["timeout"] = float(os.getenv("OPENROUTER_TIMEOUT", "180"))  # type: ignore
            self.client = openai.OpenAI(**client_kwargs)
        else:
            raise ValueError(
                "Unsupported model source. Supported sources are: openai, openrouter, vllm."
            )

    def init_chat(
        self,
        task_prompt,
    ):
        self.history = [
            {"role": "system", "content": task_prompt},
        ]
        self.reasoning_trace = []  # Reset reasoning trace when starting new chat

    def send_message(
        self,
        message,
        max_new_tokens=None,
        truncate_history=False,
        cot=False,
        image_only=False,
        stream: bool = False,
        allow_partial_on_error: bool = True,
        continue_on_truncation: bool = False,
        max_continuations: int = 0,
    ):
        if not validate_message_turns(self.history):
            raise ValueError(
                "Invalid message turn sequence detected. Check error.json for details."
            )

        # Store original response
        raw_response = None
        raw_reasoning = None
        if self.image_input:
            image_file_path = os.path.join(self.image_path, "current.png")
            base64_image = encode_image_to_base64(image_file_path)
            content = []
            if message and not image_only:
                if "qwen3" in self.model_name and not cot:
                    message += "\n/no_think"
                content.append({"type": "text", "text": message})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image,
                    },
                }
            )
            self.history.append(
                {
                    "role": "user",
                    "content": content,
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

        # Metadata placeholders
        self.last_finish_reason = None
        self.last_truncated = False

        # Unified retry logic with optional streaming
        attempts = self.n_retry + 1  # total attempts including first try
        partial_buffer = ""
        for attempt in range(1, attempts + 1):
            try:
                if self.model_source == "openai":
                    if stream:
                        stream_resp = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=self.history,
                            max_completion_tokens=max_new_tokens or self.max_new_tokens,
                            extra_body=extra_body,
                            stream=True,
                        )
                        for chunk in stream_resp:  # type: ignore
                            try:
                                delta = chunk.choices[0].delta
                                if hasattr(delta, "content") and delta.content:
                                    partial_buffer += delta.content
                            except Exception:
                                pass
                            fr = getattr(chunk.choices[0], "finish_reason", None)
                            if fr:
                                self.last_finish_reason = fr
                        raw_response = partial_buffer
                        raw_reasoning = None
                    else:
                        raw_resp = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=self.history,
                            max_completion_tokens=max_new_tokens or self.max_new_tokens,
                            extra_body=extra_body,
                        )
                        raw_response = raw_resp.choices[0].message.content
                        self.last_finish_reason = getattr(
                            raw_resp.choices[0], "finish_reason", None
                        )
                        raw_reasoning = getattr(
                            raw_resp.choices[0].message, "reasoning", None
                        )
                else:  # openrouter or vllm
                    if stream:
                        stream_resp = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=self.history,
                            max_tokens=max_new_tokens or self.max_new_tokens,
                            temperature=0.0,
                            extra_body=extra_body,
                            stream=True,
                        )
                        for chunk in stream_resp:  # type: ignore
                            try:
                                # Prefer OpenAI-compatible delta structure
                                delta = getattr(chunk.choices[0], "delta", None)
                                text = None
                                if delta is not None and hasattr(delta, "content"):
                                    text = delta.content
                                # Some servers might stream message.content directly
                                if not text:
                                    msg = getattr(chunk.choices[0], "message", None)
                                    if msg is not None and hasattr(msg, "content"):
                                        text = msg.content
                                if text:
                                    partial_buffer += text
                            except Exception:
                                pass
                            fr = getattr(chunk.choices[0], "finish_reason", None)
                            if fr:
                                self.last_finish_reason = fr
                        raw_response = partial_buffer
                        raw_reasoning = None
                    else:
                        raw_resp = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=self.history,
                            max_tokens=max_new_tokens or self.max_new_tokens,
                            temperature=0.0,
                            # top_p=0.95,
                            extra_body=extra_body,
                        )
                        fr = getattr(raw_resp.choices[0], "finish_reason", None)
                        self.last_finish_reason = fr
                        if fr == "error":
                            print(
                                f"Model returned error: {raw_resp.choices[0].message.content}"
                            )
                        raw_response = raw_resp.choices[0].message.content
                        raw_reasoning = getattr(
                            raw_resp.choices[0].message, "reasoning", None
                        )

                # Detect truncation by finish_reason
                if self.last_finish_reason in {"length", "max_tokens"}:
                    self.last_truncated = True
                break  # success
            except Exception as e:
                if stream and allow_partial_on_error and partial_buffer:
                    print(
                        f"Streaming error, returning partial after attempt {attempt}: {e}"
                    )
                    raw_response = partial_buffer
                    raw_reasoning = None
                    break
                if attempt == attempts:
                    source = "OpenAI" if self.model_source == "openai" else "API"
                    print(f"{source} error after {attempt} attempts: {e}")
                    if allow_partial_on_error and partial_buffer:
                        print("Returning partial buffer instead of None.")
                        raw_response = partial_buffer
                        raw_reasoning = None
                        break
                    return None
                # Backoff: smaller for openai (keep original 5s), larger for others (original 60s)
                sleep_time = 5 if self.model_source == "openai" else 60
                time.sleep(sleep_time)
                continue

        # Optionally attempt continuation calls if truncated and allowed
        continuation_count = 0
        while (
            continue_on_truncation
            and self.last_truncated
            and continuation_count < max_continuations
            and raw_response is not None
        ):
            continuation_count += 1
            # Append a simple continuation instruction; could be configurable
            self.history.append({"role": "assistant", "content": raw_response})
            self.history.append(
                {
                    "role": "user",
                    "content": "Continue the previous answer. Only continue; do not repeat earlier text.",
                }
            )
            try:
                cont_resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.history,
                    **(
                        {"max_completion_tokens": max_new_tokens or self.max_new_tokens}
                        if self.model_source == "openai"
                        else {
                            "max_tokens": max_new_tokens or self.max_new_tokens,
                            "temperature": 0.0,
                        }
                    ),
                )
                cont_text = cont_resp.choices[0].message.content
                fr = getattr(cont_resp.choices[0], "finish_reason", None)
                self.last_finish_reason = fr
                if fr in {"length", "max_tokens"}:
                    self.last_truncated = True
                else:
                    self.last_truncated = False
                raw_response += cont_text if cont_text else ""
            except Exception as e:
                print(f"Continuation attempt {continuation_count} failed: {e}")
                break

        # Add this code after getting raw_response but before updating history
        if cot:
            if raw_reasoning:
                self.reasoning_trace.append(
                    {"user_message": message, "reasoning": raw_reasoning.strip()}
                )
            else:
                # Extract reasoning trace from response
                trace = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
                if not trace:
                    trace = re.search(
                        r"<thinking>(.*?)</thinking>", raw_response, re.DOTALL
                    )
                if trace:
                    self.reasoning_trace.append(
                        {"user_message": message, "reasoning": trace.group(1).strip()}
                    )
                else:
                    self.reasoning_trace.append(
                        {"user_message": message, "reasoning": raw_response.strip()}
                    )

        response = raw_response
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

    # --------------------------- Batch API helpers --------------------------- #
    def batch_create_requests(self, request_specs, jsonl_path: str):
        """Create a JSONL file for OpenAI Batch API.

        Args:
            request_specs: iterable of dicts with keys:
               custom_id (str) unique
               messages (list) OpenAI chat messages
               max_completion_tokens (int) optional
               temperature (float) optional
               extra_body (dict) optional
            jsonl_path: destination path for requests file

        Note: Only supported for model_source == 'openai'.
        """
        if self.model_source != "openai":
            raise ValueError("Batch API currently only supported for OpenAI source")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for spec in request_specs:
                body = {
                    "model": self.model_name,
                    "messages": spec["messages"],
                    "max_completion_tokens": spec.get(
                        "max_completion_tokens", self.max_new_tokens
                    ),
                }
                # Merge any additional provided body keys (e.g., reasoning)
                extra = spec.get("extra_body") or {}
                if extra:
                    body.update(extra)
                line = {
                    "custom_id": spec["custom_id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(line) + "\n")

    def batch_submit(
        self,
        jsonl_path: str,
        completion_window: str = "24h",
        metadata: dict | None = None,
    ):
        """Submit a previously created JSONL requests file to OpenAI Batch API.

        Returns: batch id (str)
        """
        if self.model_source != "openai":
            raise ValueError("Batch API currently only supported for OpenAI source")
        with open(jsonl_path, "rb") as f:
            upload = self.client.files.create(file=f, purpose="batch")  # type: ignore
        batch = self.client.batches.create(  # type: ignore
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata=metadata or {},
        )
        return batch.id

    def batch_status(self, batch_id: str):
        if self.model_source != "openai":
            raise ValueError("Batch API currently only supported for OpenAI source")
        return self.client.batches.retrieve(batch_id)  # type: ignore

    def batch_collect(self, batch_id: str, output_jsonl_path: str | None = None):
        """Attempt to collect batch responses.

        If the batch is not yet completed, returns a status summary dict instead
        of waiting. This avoids blocking when the user just wants progress.

        Returns:
            list[dict]    -> when completed (parsed JSONL lines)
            dict          -> when not completed or failed, keys include:
                              status, processed, pending, total, failed, message
        """
        if self.model_source != "openai":
            raise ValueError("Batch API currently only supported for OpenAI source")

        batch = self.client.batches.retrieve(batch_id)  # type: ignore
        print(batch)
        status = getattr(batch, "status", None)
        # Attempt to read request counts (structure may vary)
        counts = getattr(batch, "request_counts", {}) or {}

        def _c(obj, *names):
            for n in names:
                if isinstance(obj, dict) and n in obj:
                    return obj[n] or 0
                if hasattr(obj, n):
                    try:
                        val = getattr(obj, n)
                        if val is not None:
                            return val
                    except Exception:
                        pass
            return 0

        total = _c(counts, "total", "requested")
        completed = _c(counts, "completed", "succeeded")
        failed = _c(counts, "failed")
        processed = completed + failed
        pending = max(total - processed, 0) if total else None

        if status != "completed":
            return {
                "status": status,
                "processed": processed,
                "failed": failed,
                "pending": pending,
                "total": total,
                "message": f"Batch {batch_id} not ready (status={status}). Processed={processed} Failed={failed} Pending={pending} Total={total}",
            }

        # Completed: extract output file id (support legacy / future variants)
        output_file_id = getattr(batch, "output_file_id", None) or getattr(
            batch, "output_file_ids", None
        )
        if isinstance(output_file_id, list):
            output_file_id = output_file_id[0] if output_file_id else None
        # If no output file, check for error file and surface its contents
        if not output_file_id:
            error_file_id = getattr(batch, "error_file_id", None) or getattr(
                batch, "error_file_ids", None
            )
            if isinstance(error_file_id, list):
                error_file_id = error_file_id[0] if error_file_id else None
            if error_file_id:
                try:
                    err_content = self.client.files.content(error_file_id)  # type: ignore
                    err_bytes = err_content.read()
                    err_text = err_bytes.decode("utf-8", errors="replace")
                    # Optionally write errors to a sidecar file if an output path was provided
                    if output_jsonl_path:
                        err_path = output_jsonl_path + ".errors"
                        with open(err_path, "w", encoding="utf-8") as ef:
                            ef.write(err_text)
                    parsed_errors = []
                    for line in err_text.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            parsed_errors.append(json.loads(line))
                        except json.JSONDecodeError:
                            parsed_errors.append({"unparsed": line})
                    return {
                        "status": status,
                        "processed": processed,
                        "failed": failed,
                        "pending": pending,
                        "total": total,
                        "errors": parsed_errors,
                        "message": f"Batch {batch_id} completed with errors only (no output file). Parsed {len(parsed_errors)} error lines.",
                    }
                except Exception as e:
                    return {
                        "status": status,
                        "processed": processed,
                        "failed": failed,
                        "pending": pending,
                        "total": total,
                        "message": f"Batch {batch_id} completed but error file retrieval failed: {e}",
                    }
            return {
                "status": status,
                "processed": processed,
                "failed": failed,
                "pending": pending,
                "total": total,
                "message": f"Batch {batch_id} completed but no output or error file id present yet.",
            }
        file_content = self.client.files.content(output_file_id)  # type: ignore
        raw_bytes = file_content.read()
        text = raw_bytes.decode("utf-8")
        if output_jsonl_path:
            with open(output_jsonl_path, "w", encoding="utf-8") as f:
                f.write(text)
        parsed = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return parsed
