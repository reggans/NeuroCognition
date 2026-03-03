from importlib.util import find_spec
from typing import Dict, Any, Union, Tuple
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def is_liger_available() -> bool:
    """Check if Liger kernel is available and explicitly enabled."""
    if os.environ.get("USE_LIGER_KERNEL", "0") != "1":
        return False
    return find_spec("liger_kernel") is not None

def is_vlm_model(model_name: str) -> bool:
    """Check if model is a vision-language model that needs special handling."""
    vlm_keywords = ["VL", "vision", "llava", "qwen2-vl", "qwen3-vl", "pixtral"]
    model_lower = model_name.lower()
    return any(kw.lower() in model_lower for kw in vlm_keywords)

def _get_default_attn_impl() -> str:
    if os.environ.get("USE_FLASH_ATTN", "0") == "1":
        try:
            import importlib

            importlib.import_module("flash_attn")
            return "flash_attention_2"
        except Exception:
            return "sdpa"
    return "sdpa"


def get_model(model_name: str, model_kwargs: Union[Dict[str, Any], None] = None) -> Any:
    if model_kwargs is None:
        model_kwargs = dict(
            dtype=torch.bfloat16,
            attn_implementation=_get_default_attn_impl(),
        )
    
    # Check if this is a VLM model
    if is_vlm_model(model_name):
        print(f"Detected VLM model: {model_name}, using AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        # VLM models don't accept use_cache in constructor
        # Use sdpa instead of flash_attention_2 due to binary compatibility issues
        vlm_kwargs = {k: v for k, v in model_kwargs.items() if k != "use_cache"}
        vlm_kwargs["attn_implementation"] = "sdpa"
        return AutoModelForImageTextToText.from_pretrained(model_name, **vlm_kwargs)
    
    # For non-VLM models, optionally try Liger kernel (disabled by default)
    if is_liger_available():
        print("Using Liger kernel")
        from liger_kernel.transformers import AutoLigerKernelForCausalLM  # type: ignore
        return AutoLigerKernelForCausalLM.from_pretrained(model_name, **model_kwargs)

    return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
def get_tokenizer(model_name: str) -> Any:
    """Get tokenizer/processor for the model."""
    # For VLM models, we need AutoProcessor
    if is_vlm_model(model_name):
        print(f"Using AutoProcessor for VLM model: {model_name}")
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return processor
    
    tokenizer = None
    if "Instruct" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name + "-Instruct")
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not hasattr(tokenizer, "chat_template"):
        raise ValueError(f"Tokenizer for model {model_name} does not have chat_template attribute, \
                            and could not find a tokenizer with the same name as the model with suffix \
                            '-Instruct'. Please provide a tokenizer with the chat_template attribute.")
    return tokenizer
            
def get_model_and_tokenizer(model_name: str, model_kwargs: Union[Dict[str, Any], None] = None) -> Tuple[Any, Any]:
    model = get_model(model_name, model_kwargs)
    tokenizer = get_tokenizer(model_name)
    return model, tokenizer