import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from collections import OrderedDict
import numpy as np


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiframeIntegrationTransformer(nn.Module):
    def __init__(self, T, embed_dim=512, layers=1,):
        super().__init__()
        self.T = T
        transformer_heads = embed_dim // 64
        self.positional_embedding = nn.Parameter(torch.empty(1, T, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(d_model=embed_dim, n_head=transformer_heads) for _ in range(layers)])

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        ori_x = x
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  
        x = x.type(ori_x.dtype) + ori_x
        
        return x.mean(dim=1, keepdim=False)
    
    
    
    
class DinoV2_A(torch.nn.Module):
    def __init__(self, backbone_arch = 'dinov2_vits14', T = 5, mit_layers = 2, freeze_backbone=True):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', backbone_arch)
        if backbone_arch == 'dinov2_vits14' or backbone_arch == 'dinov2_vits14_reg':
            embed_dim = 384
        elif backbone_arch == 'dinov2_vitb14' or backbone_arch == 'dinov2_vitb14_reg':
            embed_dim = 768
        elif backbone_arch == 'dinov2_vitl14' or backbone_arch == 'dinov2_vitl14_reg':
            embed_dim = 1024
        elif backbone_arch == 'dinov2_vitg14' or backbone_arch == 'dinov2_vitg14_reg':
            embed_dim = 1280
        self.head_video = nn.Linear(embed_dim, embed_dim)
        self.u_head_video = nn.Linear(embed_dim, embed_dim)
        self.mit = MultiframeIntegrationTransformer(T=T, embed_dim=embed_dim, layers=mit_layers)
        self.freeze_backbone = freeze_backbone
        self.classification_head = torch.nn.Sequential(
            QuickGELU(),
            torch.nn.Linear(embed_dim, 2))
        self.classification_head_u = torch.nn.Sequential(
            QuickGELU(),
            torch.nn.Linear(embed_dim, 2))
    def forward_features(self, x):
        x = self.backbone(x)
        return x
    
    def encode_image(self, image):
    
        return self.forward_features(image)
    
    def encode_video(self, image):
        b,t,c,h,w = image.size()
        image = image.reshape(-1,c,h,w)

        video_features = self.encode_image(image)
        video_features = video_features.view(b, t, -1)
        

        video_features = self.mit(video_features)
        return video_features
    
    
    def uda(self, video_feature, train_flag):
        v_fea = self.head_video(video_feature)
        v_fea_u = self.u_head_video(video_feature)
        v_fea = v_fea / v_fea.norm(dim=-1, keepdim=True)
        v_fea_u = v_fea_u / v_fea_u.norm(dim=-1, keepdim=True)

        if train_flag:
            v_fea_u_nograd = self.u_head_video(video_feature) # was detached before
            return video_feature, v_fea, v_fea_u, None, v_fea_u_nograd, v_fea_u_nograd, None
        else:
            return video_feature, v_fea, v_fea_u, None
        
        
    def forward(self, image):
        video_features = self.encode_video(image)
        if self.training:
            _, v_features, v_features_u, t_features, \
                _, v_features_u_n, t_features_n = self.uda(video_features, self.training)
            logits = self.classification_head(v_features)
            logits_u = self.classification_head_u(v_features_u)
            logits_u_n = v_features_u_n

            outputs=  {
                    "y": logits,
                    "y_cluster_all": logits_u,
                    "feature_v": video_features,
                    "y_cluster_all_nograd": logits_u_n
                }
            return outputs
        else:
            video_features, v_features, v_features_u, t_features= self.uda(video_features, self.training)
            logits = self.classification_head(v_features)
            logits_u = self.classification_head_u(v_features_u)
            outputs = {
                "y": logits,
                "y_cluster_all": logits_u,
                "feature_v": video_features,
            }

            return outputs
        
        