# import google.generativeai as genai
import google
import openai
import transformers
from huggingface_hub import login
from openai import OpenAI
# from vllm import LLM
import torch
from tqdm.auto import tqdm

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
        
    for i in range(start_idx, len(messages)-1):
        current_role = messages[i]["role"]
        next_role = messages[i+1]["role"]
        
        # Check if same role appears consecutively
        if current_role == next_role:
            if save_error:
                error_info = {
                    "error": "Non-alternating message turns detected",
                    "position": i,
                    "messages": messages
                }
                with open("error.json", "w") as f:
                    json.dump(error_info, f, indent=2)
            return False
            
        # Verify valid role pairs
        valid_pairs = {
            "user": ["assistant", "model"],
            "assistant": ["user"],
            "model": ["user"]
        }
        
        if next_role not in valid_pairs.get(current_role, []):
            if save_error:
                error_info = {
                    "error": f"Invalid role sequence: {current_role} -> {next_role}",
                    "position": i,
                    "messages": messages
                }
                with open("error.json", "w") as f:
                    json.dump(error_info, f, indent=2)
            return False
            
    return True

class ModelWrapper():
    def __init__(self, model_name, model_source, api_key=None, max_new_tokens=512):
        self.chat = None # Specific to Gemini API, None otherwise
        self.client = None # Specific to LiteLLM API, None otherwise
        self.history = None
        self.model_name = model_name
        self.model_source = model_source
        self.max_new_tokens = max_new_tokens
        self.reasoning_trace = []  # Add new private attribute

        # Gemini API
        if model_source == "google":
            if api_key is None:
                if os.getenv("GOOGLE_API_KEY") is not None:
                    api_key = os.getenv("GOOGLE_API_KEY")
                else:
                    raise ValueError("Please set the GOOGLE_API_KEY environment variable or pass it to the CLI.")
            # genai.configure(api_key=api_key)

            # self.model = genai.GenerativeModel(model_name)
            self.tokenizer = None
        # Get model from Hugging Face
        elif model_source == "hf":
            if api_key is None:
                if os.getenv("HF_TOKEN") is not None:
                    api_key = os.getenv("HF_TOKEN") 
                else:
                    raise ValueError("Please set the HF_TOKEN environment variable or pass it to the CLI.")
            login(token=api_key)

            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name,
                                                                           torch_dtype="auto",
                                                                           device_map="auto")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        # LiteLLM API
        elif model_source == "litellm":
            if api_key is None:
                if os.getenv("LITELLM_API_KEY") is not None:
                    api_key = os.getenv("LITELLM_API_KEY")
                else:
                    raise ValueError("Please set the LITELLM_API_KEY environment variable or pass it to the CLI.")

            self.client = openai.OpenAI(
                api_key=api_key,
                base_url = "https://litellm.rum.uilab.kr:8080/"
            )
        # VLLM API
        elif model_source == "vllm":
            vllm_url = f"http://{os.getenv('VLLM_URL')}:8877/v1"
            self.client = OpenAI(
                base_url=vllm_url,
                api_key="dummy"  # VLLM doesn't need real API key
            )
            self.model_name = model_name
    
    def init_chat(self, task_prompt):
        if self.model_source == "google":
            # Initialize history
            self.history = [
                {"role": "user", "content": task_prompt},
                {"role": "model", "content": "Understood, lets start."},
            ]
            
            # Start Gemini chat
            self.chat = self.model.start_chat(
                history = self.history,
            )
        else:
            self.history = [
                {"role": "system", "content": task_prompt},
            ]
        
        self.reasoning_trace = []  # Reset reasoning trace when starting new chat
        
    def send_message(self, message, max_new_tokens=None, truncate_history=False, cot=False):
        if not validate_message_turns(self.history):
            raise ValueError("Invalid message turn sequence detected. Check error.json for details.")

        # Store original response
        raw_response = None

        self.history.append(
            {"role": "user", "content": message}
        )

        if self.model_source == "google":
            try:
                raw_response = self.chat.send_message(message).text
            except google.api_core.exceptions.ResourceExhausted:
                time.sleep(60)
                raw_response = self.chat.send_message(message).text

        elif self.model_source == "hf":
            if max_new_tokens is None:
                max_new_tokens = self.max_new_tokens
                
            for i in range(3):
                if re.search(r"<answer>(?s:.*)</answer>", raw_response) is not None:
                    break
                # Continue answer
                text = self.tokenizer.apply_chat_template(
                    self.history,
                    tokenize=False,
                    continue_final_message=True
                )

                model_inputs = self.tokenizer([text], 
                                              return_tensors="pt",
                                              truncation=True, 
                                              max_length=120000).to(self.model.device)
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                raw_response += self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            tqdm.write(raw_response)
            
            model_inputs = None 
            generated_ids = None
            torch.cuda.empty_cache()
        
        elif self.model_source in ["litellm", "vllm"]:
            try:
                raw_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.history,
                    max_tokens=max_new_tokens or self.max_new_tokens,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": bool(cot)},
                    },
                )
            except:
                time.sleep(5)
                raw_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.history,
                    max_tokens=max_new_tokens or self.max_new_tokens,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": bool(cot)},
                    },
                )
            raw_response = raw_response.choices[0].message.content

        # Add this code after getting raw_response but before updating history
        if cot:
            # Extract reasoning trace from response
            trace = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
            if trace:
                self.reasoning_trace.append({
                    "user_message": message,
                    "reasoning": trace.group(1).strip()
                })
            else:
                self.reasoning_trace.append({
                    "user_message": message,
                    "reasoning": raw_response.strip()
                })

        # Parse response
        if truncate_history:
            # Remove reasoning trace and get content after </think>
            parsed = re.search(r"</think>(.*?)$", raw_response, re.DOTALL)
            if parsed:
                response = parsed.group(1).strip()
            else:
                 # If no closing </think> tag found, limit to last 256 words
                words = raw_response.split()
                response = ' '.join(words[-256:]).strip()
        
        # Update history based on model source
        if self.model_source == "google":
            self.history.extend([
                {"role": "user", "content": message},
                {"role": "model", "content": response if truncate_history else raw_response or response}
            ])
        elif self.model_source == "hf":
            if response:  
                self.history.append({"role": "model", "content": response if truncate_history else raw_response or response})
            else:
                self.history.append({"role": "model", "content": response})
        elif self.model_source in ["litellm", "vllm"]:
            self.history.append({"role": "assistant", "content": response if truncate_history else raw_response or response})

        return raw_response
