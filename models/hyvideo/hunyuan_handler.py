import torch

def get_hunyuan_text_encoder_filename(text_encoder_quantization):
    if text_encoder_quantization =="int8":
        text_encoder_filename = "ckpts/llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_quanto_int8.safetensors"
    else:
        text_encoder_filename = "ckpts/llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_fp16.safetensors"

    return text_encoder_filename

class family_handler():
    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {}

        if base_model_type in ["hunyuan_avatar", "hunyuan_custom_audio"]:
            fps = 25
        elif base_model_type in ["hunyuan", "hunyuan_i2v", "hunyuan_custom_edit", "hunyuan_custom"]:
            fps = 24
        else:
            fps = 16
        extra_model_def["fps"] = fps
        extra_model_def["frames_minimum"] = 5
        extra_model_def["frames_steps"] = 4
        extra_model_def["sliding_window"] = False
        extra_model_def["embedded_guidance"] = base_model_type in ["hunyuan", "hunyuan_i2v"]
        extra_model_def["cfg_star"] =  base_model_type in [ "hunyuan_avatar", "hunyuan_custom_audio", "hunyuan_custom_edit", "hunyuan_custom"]
        extra_model_def["skip_steps_cache"] = True
        return extra_model_def

    @staticmethod
    def query_supported_types():
        return ["hunyuan", "hunyuan_i2v", "hunyuan_custom", "hunyuan_custom_audio", "hunyuan_custom_edit", "hunyuan_avatar"]

    @staticmethod
    def query_family_maps():
        models_eqv_map = {
        }

        models_comp_map = { 
                    "hunyuan_custom":  ["hunyuan_custom_edit", "hunyuan_custom_audio"],
                    }

        return models_eqv_map, models_comp_map

    @staticmethod
    def query_model_family():
        return "hunyuan"

    @staticmethod
    def query_family_infos():
        return {"hunyuan":(20, "Hunyuan Video")}

    @staticmethod
    def get_rgb_factors(model_type):
        from shared.RGB_factors import get_rgb_factors
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("hunyuan")
        return latent_rgb_factors, latent_rgb_factors_bias

    @staticmethod
    def query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization):
        text_encoder_filename = get_hunyuan_text_encoder_filename(text_encoder_quantization)    
        return {  
            "repoId" : "DeepBeepMeep/HunyuanVideo", 
            "sourceFolderList" :  [ "llava-llama-3-8b", "clip_vit_large_patch14",  "whisper-tiny" , "det_align", ""  ],
            "fileList" :[ ["config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"] + computeList(text_encoder_filename) ,
                            ["config.json", "merges.txt", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
                            ["config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer_config.json"],
                            ["detface.pt"],
                            [ "hunyuan_video_720_quanto_int8_map.json", "hunyuan_video_custom_VAE_fp32.safetensors", "hunyuan_video_custom_VAE_config.json", "hunyuan_video_VAE_fp32.safetensors", "hunyuan_video_VAE_config.json" , "hunyuan_video_720_quanto_int8_map.json"   ] + computeList(model_filename)  
                            ]
        } 

    @staticmethod
    def load_model(model_filename, model_type = None,  base_model_type = None, model_def = None, quantizeTransformer = False, text_encoder_quantization = None, dtype = torch.bfloat16, VAE_dtype = torch.float32, mixed_precision_transformer = False, save_quantized = False):
        from .hunyuan import HunyuanVideoSampler
        from mmgp import offload

        hunyuan_model = HunyuanVideoSampler.from_pretrained(
            model_filepath = model_filename,
            model_type = model_type, 
            base_model_type = base_model_type,
            text_encoder_filepath = get_hunyuan_text_encoder_filename(text_encoder_quantization),
            dtype = dtype,
            quantizeTransformer = quantizeTransformer,
            VAE_dtype = VAE_dtype, 
            mixed_precision_transformer = mixed_precision_transformer,
            save_quantized = save_quantized
        )

        pipe = { "transformer" : hunyuan_model.model, "text_encoder" : hunyuan_model.text_encoder, "text_encoder_2" : hunyuan_model.text_encoder_2, "vae" : hunyuan_model.vae  }

        if hunyuan_model.wav2vec != None:
            pipe["wav2vec"] = hunyuan_model.wav2vec


        # if hunyuan_model.align_instance != None:
        #     pipe["align_instance"] = hunyuan_model.align_instance.facedet.model


        from .modules.models import get_linear_split_map

        split_linear_modules_map = get_linear_split_map()
        hunyuan_model.model.split_linear_modules_map = split_linear_modules_map
        offload.split_linear_modules(hunyuan_model.model, split_linear_modules_map )


        return hunyuan_model, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults["embedded_guidance_scale"]= 6.0

        if base_model_type in ["hunyuan","hunyuan_i2v"]:
            ui_defaults.update({
                "guidance_scale": 7.0,
            })
    
        elif base_model_type in ["hunyuan_custom"]:
            ui_defaults.update({
                "guidance_scale": 7.5,
                "flow_shift": 13,
                "resolution": "1280x720",
                "video_prompt_type": "I",
            })
        elif base_model_type in ["hunyuan_custom_audio"]:
            ui_defaults.update({
                "guidance_scale": 7.5,
                "flow_shift": 13,
                "video_prompt_type": "I",
            })
        elif base_model_type in ["hunyuan_custom_edit"]:
            ui_defaults.update({
                "guidance_scale": 7.5,
                "flow_shift": 13,
                "video_prompt_type": "MVAI",
                "sliding_window_size": 129,
            })
        elif base_model_type in ["hunyuan_avatar"]:
            ui_defaults.update({
                "guidance_scale": 7.5,
                "flow_shift": 5,
                "remove_background_images_ref": 0,
                "skip_steps_start_step_perc": 25, 
                "video_length": 129,
                "video_prompt_type": "I",
            })
