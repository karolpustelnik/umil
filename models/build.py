from models.xclip import XCLIP
from typing import Tuple, Union
import torch
import sys
import warnings
sys.path.append("../")
import clip
from models.dino_v2 import DinoV2_A

def build_model(state_dict: dict, T=8, droppath=0., use_checkpoint=False, logger=None, prompts_alpha=1e-1, prompts_layers=2, use_cache=True, 
                mit_layers=4, input_channels = 3, backbone_arch = 'dinov2_vits14', freeze_backbone=True, model_type = 'clip'):
    if model_type == 'clip':
        vit = "visual.proj" in state_dict

        if vit:
            vision_width = state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            
            vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        model = XCLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,  
            T=T, droppath=droppath, mit_layers=mit_layers,
            prompts_alpha=prompts_alpha, prompts_layers=prompts_layers,
            use_checkpoint=use_checkpoint, use_cache=use_cache, input_channels=input_channels
        )

        for key in ["input_resolution", "context_length", "vocab_size", "mit.positional_embedding"]:
            if key in state_dict:
                del state_dict[key]
        if input_channels != 3:
            if 'visual.conv1.weight' in state_dict:
                state_dict['visual.conv1.weight'] = model.visual.conv1.weight
        msg = model.load_state_dict(state_dict,strict=False)
        # freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        #unfreeze selected layers
        for param in model.visual.conv1.parameters():
            param.requires_grad = True
            
        for param in model.head_video.parameters():
            param.requires_grad = True
        for param in model.u_head_video.parameters():
            param.requires_grad = True
        logger.info(f"load pretrained CLIP: {msg}")
        
    elif model_type == 'dino':
        model = DinoV2_A(backbone_arch = backbone_arch, T = T, mit_layers = mit_layers, freeze_backbone=freeze_backbone)
        
    return model.eval()


def load_model(model_path, name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", 
         jit=True, T=8, droppath=0., use_checkpoint=False, logger=None, use_cache=True, prompts_alpha=1e-1, 
         prompts_layers=2, mit_layers=1, input_channels = 3, backbone_arch = 'dinov2_vits14', freeze_backbone = True, model_type = 'clip'):
    if model_type == 'clip':
        if model_path is None:
            model_path = clip._download(clip._MODELS[name])

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(model_path, map_location="cpu")

        model = build_model(state_dict['model'] or model.state_dict(), T=T, droppath=droppath,
                            use_checkpoint=use_checkpoint, logger=logger,
                            prompts_alpha=prompts_alpha, 
                            prompts_layers=prompts_layers,
                            use_cache=use_cache,
                            mit_layers=mit_layers,
                            input_channels = input_channels, 
                            backbone_arch=backbone_arch,
                            freeze_backbone = freeze_backbone,
                            model_type = model_type
                            )
        if str(device) == "cpu":
            model.float()
    elif model_type == 'dino':
        model = build_model(None, T=T, droppath=droppath,
                            use_checkpoint=use_checkpoint, logger=logger,
                            prompts_alpha=prompts_alpha, 
                            prompts_layers=prompts_layers,
                            use_cache=use_cache,
                            mit_layers=mit_layers,
                            input_channels = input_channels,
                            backbone_arch=backbone_arch,
                            freeze_backbone = freeze_backbone,
                            model_type = model_type,)
        if str(device) == "cpu":
            model.float()
    return model, model.state_dict()