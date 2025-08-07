import torch

def test_class_i2v(base_model_type):    
    return base_model_type in ["i2v", "i2v_2_2", "fun_inp_1.3B", "fun_inp", "flf2v_720p",  "fantasy",  "multitalk",  ] #"hunyuan_i2v",

class family_handler():
    @staticmethod
    def get_wan_text_encoder_filename(text_encoder_quantization):
        text_encoder_filename = "ckpts/umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors"
        if text_encoder_quantization =="int8":
            text_encoder_filename = text_encoder_filename.replace("bf16", "quanto_int8") 
        return text_encoder_filename



    @staticmethod
    def query_modules_files():
        return {
            "vace_14B" : ["ckpts/wan2.1_Vace_14B_module_mbf16.safetensors", "ckpts/wan2.1_Vace_14B_module_quanto_mbf16_int8.safetensors", "ckpts/wan2.1_Vace_14B_module_quanto_mfp16_int8.safetensors"],
            "vace_1.3B" : ["ckpts/wan2.1_Vace_1_3B_module.safetensors"],
            "fantasy": ["ckpts/wan2.1_fantasy_speaking_14B_bf16.safetensors"],
            "multitalk": ["ckpts/wan2.1_multitalk_14B_mbf16.safetensors", "ckpts/wan2.1_multitalk_14B_quanto_mbf16_int8.safetensors", "ckpts/wan2.1_multitalk_14B_quanto_mfp16_int8.safetensors"]
}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {}
        if "URLs2" in model_def:
            extra_model_def["no_steps_skipping"] = True
        extra_model_def["i2v_class"] = test_class_i2v(base_model_type)

        vace_class = base_model_type in ["vace_14B", "vace_1.3B", "vace_multitalk_14B"] 
        extra_model_def["vace_class"] = vace_class

        if base_model_type in ["multitalk", "vace_multitalk_14B"]:
            fps = 25
        elif base_model_type in ["fantasy"]:
            fps = 23
        elif base_model_type in ["ti2v_2_2"]:
            fps = 24
        else:
            fps = 16
        extra_model_def["fps"] =fps

        if vace_class: 
            frames_minimum, frames_steps =  17, 4
        else:
            frames_minimum, frames_steps = 5, 4
        extra_model_def.update({
        "frames_minimum" : frames_minimum,
        "frames_steps" : frames_steps, 
        "sliding_window" : base_model_type in ["multitalk", "t2v", "fantasy"] or test_class_i2v(base_model_type) or vace_class,  #"ti2v_2_2",
        "guidance_max_phases" : 2,
        "skip_layer_guidance" : True,        
        "cfg_zero" : True,
        "cfg_star" : True,
        "adaptive_projected_guidance" : True,  
        "skip_steps_cache" : not (base_model_type in ["i2v_2_2", "ti2v_2_2" ] or "URLs2" in model_def),
        })

        return extra_model_def
        
    @staticmethod
    def query_supported_types():
        return ["multitalk", "fantasy", "vace_14B", "vace_multitalk_14B",
                    "t2v_1.3B", "t2v", "vace_1.3B", "phantom_1.3B", "phantom_14B", 
                    "recam_1.3B", 
                    "i2v", "i2v_2_2", "ti2v_2_2", "flf2v_720p", "fun_inp_1.3B", "fun_inp"]


    @staticmethod
    def query_family_maps():

        models_eqv_map = {
            "flf2v_720p" : "i2v",
            "t2v_1.3B" : "t2v",
        }

        models_comp_map = { 
                    "vace_14B" : [ "vace_multitalk_14B"],
                    "t2v" : [ "vace_14B", "vace_1.3B" "vace_multitalk_14B", "t2v_1.3B", "phantom_1.3B","phantom_14B"],
                    "i2v" : [ "fantasy", "multitalk", "flf2v_720p" ],
                    "fantasy": ["multitalk"],
                    }
        return models_eqv_map, models_comp_map

    @staticmethod
    def query_model_family():
        return "wan"
    
    @staticmethod
    def query_family_infos():
        return {"wan":(0, "Wan2.1"), "wan2_2":(1, "Wan2.2") }

    @staticmethod
    def get_vae_block_size(base_model_type):
        return 32 if base_model_type == "ti2v_2_2" else 16

    @staticmethod
    def get_rgb_factors(model_type):
        from shared.RGB_factors import get_rgb_factors
        if model_type == "ti2v_2_2": return None, None
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("wan")
        return latent_rgb_factors, latent_rgb_factors_bias
    
    @staticmethod
    def query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization):
        text_encoder_filename = family_handler.get_wan_text_encoder_filename(text_encoder_quantization)

        download_def  = [{
            "repoId" : "DeepBeepMeep/Wan2.1", 
            "sourceFolderList" :  ["xlm-roberta-large", "umt5-xxl", ""  ],
            "fileList" : [ [ "models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors", "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"], ["special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"] + computeList(text_encoder_filename) , ["Wan2.1_VAE.safetensors",  "fantasy_proj_model.safetensors" ] +  computeList(model_filename)  ]   
        }]

        if base_model_type == "ti2v_2_2":
            download_def += [    {
                "repoId" : "DeepBeepMeep/Wan2.2", 
                "sourceFolderList" :  [""],
                "fileList" : [ [ "Wan2.2_VAE.safetensors" ]  ]
            }]

        return download_def


    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer = False, text_encoder_quantization = None, dtype = torch.bfloat16, VAE_dtype = torch.float32, mixed_precision_transformer = False, save_quantized= False):
        from .configs import WAN_CONFIGS

        if test_class_i2v(base_model_type):
            cfg = WAN_CONFIGS['i2v-14B']
        else:
            cfg = WAN_CONFIGS['t2v-14B']
            # cfg = WAN_CONFIGS['t2v-1.3B']    
        from . import WanAny2V
        wan_model = WanAny2V(
            config=cfg,
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type = model_type,        
            model_def = model_def,
            base_model_type=base_model_type,
            text_encoder_filename= family_handler.get_wan_text_encoder_filename(text_encoder_quantization),
            quantizeTransformer = quantizeTransformer,
            dtype = dtype,
            VAE_dtype = VAE_dtype, 
            mixed_precision_transformer = mixed_precision_transformer,
            save_quantized = save_quantized
        )

        pipe = {"transformer": wan_model.model, "text_encoder" : wan_model.text_encoder.model, "vae": wan_model.vae.model }
        if hasattr(wan_model,"model2") and wan_model.model2 is not None:
            pipe["transformer2"] = wan_model.model2
        if hasattr(wan_model, "clip"):
            pipe["text_encoder_2"] = wan_model.clip.model
        return wan_model, pipe
    
    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        if base_model_type in ["fantasy"]:
            ui_defaults.update({
                "audio_guidance_scale": 5.0,
                "sliding_window_size": 1, 
            })

        elif base_model_type in ["multitalk"]:
            ui_defaults.update({
                "guidance_scale": 5.0,
                "flow_shift": 7, # 11 for 720p
                "audio_guidance_scale": 4,
                "sliding_window_discard_last_frames" : 4,
                "sample_solver" : "euler",
                "adaptive_switch" : 1,
            })

        elif base_model_type in ["phantom_1.3B", "phantom_14B"]:
            ui_defaults.update({
                "guidance_scale": 7.5,
                "flow_shift": 5,
                "remove_background_images_ref": 1,
                "video_prompt_type": "I",
                # "resolution": "1280x720" 
            })

        elif base_model_type in ["vace_14B", "vace_multitalk_14B"]:
            ui_defaults.update({
                "sliding_window_discard_last_frames": 0,
            })

        elif base_model_type in ["ti2v_2_2"]:
            ui_defaults.update({
                "image_prompt_type": "T", 
            })

            