import genai
import transformers
from huggingface_hub import login

import os

class ModelWrapper():
    def __init__(self, model_name):
        # Gemini API
        if model_name == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
            genai.configure(api_key=api_key)

            self.model = genai.GenerativeModel()
            self.tokenizer = None
        # Get model from Hugging Face
        else:
            api_key = os.getenv("HF_TOKEN")
            if api_key is None:
                raise ValueError("Please set the HF_TOKEN environment variable.")
            login(token=api_key)

            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name,
                                                                    torch_dtype="auto",
                                                                    device_map="auto")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        
        self.chat = None
        self.history = None
        self.model_name = model_name
    
    def init_chat(self, task_prompt):
        if self.model_name == "gemini":
            self.chat = self.model.start_chat(
                history = [
                    {"role": "user", "parts": task_prompt},
                    {"role": "model", "parts": "Understood, lets start the test."},
                ]
            )
        else:
            self.history = [
                {"role": "system", "content": task_prompt},
            ]
    
    def send_message(self, message, max_new_tokens=512):
        if self.model_name == "gemini":
            response = self.chat.send_message(message).text
        else:
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
                max_new_tokens=max_new_tokens
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            self.history.append({"role": "model", "content": response})
            
        return response
    
    def get_history(self):
        if self.model_name == "gemini":
            return self.chat.history
        return self.history
