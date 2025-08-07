
from mmgp import offload
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch, json, os

from diffusers.image_processor import VaeImageProcessor
from .transformer_qwenimage import QwenImageTransformer2DModel

from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, AutoTokenizer
from .autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers import FlowMatchEulerDiscreteScheduler
from .pipeline_qwenimage import QwenImagePipeline

class model_factory():
    def __init__(
        self,
        checkpoint_dir,
        model_filename = None,
        model_type = None, 
        model_def = None,
        base_model_type = None,
        text_encoder_filename = None,
        quantizeTransformer = False,
        save_quantized = False,
        dtype = torch.bfloat16,
        VAE_dtype = torch.float32,
        mixed_precision_transformer = False
    ):
        
        with open( os.path.join(checkpoint_dir, "qwen_scheduler_config.json"), 'r', encoding='utf-8') as f:
            scheduler_config = json.load(f)
        scheduler_config.pop("_class_name")
        scheduler_config.pop("_diffusers_version")

        scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)

        transformer_filename = model_filename[0]
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint_dir,"Qwen2.5-VL-7B-Instruct")) 

        with open("configs/qwen_image_20B.json", 'r', encoding='utf-8') as f:
            transformer_config = json.load(f)
        transformer_config.pop("_diffusers_version")
        transformer_config.pop("_class_name")
        transformer_config.pop("pooled_projection_dim")
        
        from accelerate import init_empty_weights
        with init_empty_weights():
            transformer = QwenImageTransformer2DModel(**transformer_config)
        offload.load_model_data(transformer, transformer_filename)
        # transformer = offload.fast_load_transformers_model("transformer_quanto.safetensors", writable_tensors= True , modelClass=QwenImageTransformer2DModel, defaultConfigPath="transformer_config.json")

        text_encoder = offload.fast_load_transformers_model(text_encoder_filename,  writable_tensors= True , modelClass=Qwen2_5_VLForConditionalGeneration,  defaultConfigPath= os.path.join(checkpoint_dir, "Qwen2.5-VL-7B-Instruct", "config.json"))
        # text_encoder = offload.fast_load_transformers_model(text_encoder_filename, do_quantize=True,  writable_tensors= True , modelClass=Qwen2_5_VLForConditionalGeneration, defaultConfigPath="text_encoder_config.json", verboseLevel=2)
        # text_encoder.to(torch.float16)
        # offload.save_model(text_encoder, "text_encoder_quanto_fp16.safetensors", do_quantize= True)

        vae = offload.fast_load_transformers_model( os.path.join(checkpoint_dir,"qwen_vae.safetensors"), writable_tensors= True , modelClass=AutoencoderKLQwenImage, defaultConfigPath=os.path.join(checkpoint_dir,"qwen_vae_config.json"))
        
        self.pipeline = QwenImagePipeline(vae, text_encoder, tokenizer, transformer, scheduler)
        self.vae=vae
        self.text_encoder=text_encoder
        self.tokenizer=tokenizer
        self.transformer=transformer
        self.scheduler=scheduler


    def generate(
        self,
        seed: int | None = None,
        input_prompt: str = "replace the logo with the text 'Black Forest Labs'",
        sampling_steps: int = 20,
        input_ref_images = None,
        width= 832,
        height=480,
        embedded_guidance_scale: float = 4,
        fit_into_canvas = None,
        callback = None,
        loras_slists = None,
        batch_size = 1,
        video_prompt_type = "",
        VAE_tile_size = None, 
        **bbargs
    ):
        # Generate with different aspect ratios
        aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472)
        }

        if VAE_tile_size is not None:
            self.vae.use_tiling  = VAE_tile_size[0] 
            self.vae.tile_latent_min_height  = VAE_tile_size[1] 
            self.vae.tile_latent_min_width  = VAE_tile_size[1]

        # width, height = aspect_ratios["16:9"]

        image = self.pipeline(
        prompt=input_prompt,
        width=width,
        height=height,
        num_inference_steps=sampling_steps,
        num_images_per_prompt = batch_size,
        true_cfg_scale=embedded_guidance_scale,
        callback = callback,
        pipeline=self,
        loras_slists=loras_slists,
        generator=torch.Generator(device="cuda").manual_seed(seed)
        )        
        if image is None: return None
        return image.transpose(0, 1)

