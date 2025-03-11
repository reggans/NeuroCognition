import google.generativeai as genai
import transformers
from huggingface_hub import login
import torch

import os

class ModelWrapper():
    def __init__(self, model_name, model_source, api_key=None, max_new_tokens=512):
        # Gemini API
        if model_source == "google":
            if api_key is None:
                if os.getenv("GOOGLE_API_KEY") is not None:
                    api_key = os.getenv("GOOGLE_API_KEY")
                else:
                    raise ValueError("Please set the GOOGLE_API_KEY environment variable or pass it to the CLI.")
            genai.configure(api_key=api_key)

            self.model = genai.GenerativeModel(model_name)
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
        
        self.chat = None # Specific to Gemini API, None otherwise
        self.history = None
        self.model_name = model_name
        self.model_source = model_source
        self.max_new_tokens = max_new_tokens
    
    def init_chat(self, task_prompt):
        if self.model_source == "google":
            # Start Gemini chat
            self.chat = self.model.start_chat(
                history = [
                    {"role": "user", "parts": task_prompt},
                    {"role": "model", "parts": "Understood, lets start."},
                ]
            )

            # Initialize history
            self.history = [
                {"role": "user", "content": task_prompt},
                {"role": "model", "content": "Understood, lets start."},
            ]
        elif self.model_source == "hf":
            self.history = [
                {"role": "system", "content": task_prompt},
            ]
    
    def send_message(self, message, max_new_tokens=None):
        if self.model_source == "google":
            response = self.chat.send_message(message).text

            # Add to history
            self.history.extend([
                {"role": "user", "content": message},
                {"role": "model", "content": response}
            ])
        elif self.model_source == "hf":
            if max_new_tokens is None:
                max_new_tokens = self.max_new_tokens
                
            self.history.append(
                {"role": "user", "content": message}
            )

            text = self.tokenizer.apply_chat_template(
                self.history,
                tokenize=False,
                add_generation_prompt=True,
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            self.history.append({"role": "model", "content": response})
            
            model_inputs = None
            generated_ids = None
            torch.cuda.empty_cache()
        return response
