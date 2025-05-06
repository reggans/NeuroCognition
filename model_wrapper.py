import google.generativeai as genai
import google
import openai
import transformers
from huggingface_hub import login
from vllm import LLM
import torch

import os, time, re

class ModelWrapper():
    def __init__(self, model_name, model_source, api_key=None, max_new_tokens=512, budget=0):
        self.chat = None # Specific to Gemini API, None otherwise
        self.client = None # Specific to LiteLLM API, None otherwise
        self.history = None
        self.model_name = model_name
        self.model_source = model_source
        self.max_new_tokens = max_new_tokens
        self.budget = budget

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
            # self.model = LLM(
            #     model=model_name,
            #     tokenizer=model_name,
            #     dtype="float16",
            #     tensor_parallel_size=8,
            # )
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
        
        elif self.model_source == "litellm":
            self.history = [
                {"role": "system", "content": task_prompt},
            ]
    
    def send_message(self, message, max_new_tokens=None):
        if self.model_source == "google":
            try:
                response = self.chat.send_message(message).text
            except google.api_core.exceptions.ResourceExhausted:
                time.sleep(60)
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
            
            response = ""
            self.history.append(
                {"role": "model", "content": response}
            )
            # while re.search(r"<answer>(?s:.*)</answer>", response) is None:
            for i in range(3):
                if i == 2:
                    self.history[-1]["content"] += ". My final answer is <answer>"

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
                response += self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # response += self.model.chat(
                #     messages=self.history,
                #     continue_final_message=True,
                # ).outputs[0].text

                # print(response)
            
            response = re.search(r"<answer>(?s:.*)</answer>", response)[0]
            self.history[-1]["content"] = response
            
            model_inputs = None 
            generated_ids = None
            torch.cuda.empty_cache()
        elif self.model_source == "litellm":
            self.history.append(
                {"role": "user", "content": message}
            )

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.history
                )
            except:
                time.sleep(60)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.history
                )
            response = response.choices[0].message.content

            self.history.append({"role": "assistant", "content": response})

        return response
