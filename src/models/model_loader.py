import torch
import logging
from transformers import (
    Qwen2_5_VLForConditionalGeneration, AutoProcessor,
    MllamaForConditionalGeneration,
    LlavaNextForConditionalGeneration
)

def is_flash_attn_compatible():
    if not torch.cuda.is_available():
        return False
    
    try:
        import flash_attn # type: ignore
        flash_available = True
    except ImportError:
        flash_available = False
    
    major = torch.cuda.get_device_properties(0).major
    return major >= 8 and flash_available # ampere, hopper gpus

USE_FLASH_ATTENTION = is_flash_attn_compatible()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Using Flash attention: {USE_FLASH_ATTENTION}")

MODEL_CONFIGS = {
    "qwen_vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_class": Qwen2_5_VLForConditionalGeneration,
        "extra_kwargs": {"attn_implementation": "flash_attention_2"} if USE_FLASH_ATTENTION else {}
    },
    "llama_vl": {
        "model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "model_class": MllamaForConditionalGeneration,
        "extra_kwargs": {"attn_implementation": "sdpa"}
    },
    "pangea_vl": {
        "model_id": "neulab/Pangea-7B-hf",
        "model_class": LlavaNextForConditionalGeneration,
        "extra_kwargs": {"attn_implementation": "flash_attention_2"} if USE_FLASH_ATTENTION else {}
    }
}

# MODEL_CONFIGS = {
#     "qwen_vl": {
#         "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
#         "model_class": Qwen2_5_VLForConditionalGeneration,
#         "extra_kwargs": {"attn_implementation": "flash_attention_2"}
#     },
#     "llama_vl": {
#         "model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
#         "model_class": MllamaForConditionalGeneration,
#         "extra_kwargs": {"attn_implementation": "sdpa"}
#     },
#     "pangea_vl": {
#         "model_id": "neulab/Pangea-7B-hf",
#         "model_class": LlavaNextForConditionalGeneration,
#         "extra_kwargs": {"attn_implementation": "flash_attention_2"}
#     }
# }

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model name '{model_name}'.")

    config = MODEL_CONFIGS[model_name]
    model_id = config["model_id"]   
    model_class = config["model_class"]
    
    model = model_class.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **config.get("extra_kwargs", {})
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    if model_name == "pangea_vl":
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True, patch_size=14)
        model.resize_token_embeddings(len(processor.tokenizer))

    return model, processor, DEVICE
