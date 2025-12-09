import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_
from torch.utils.checkpoint import checkpoint

def to_3tuple(x):
    """Convert input to 3-tuple (depth, height, width)"""
    if isinstance(x, (tuple, list)):
        if len(x) == 1:
            return (x[0], x[0], x[0])
        elif len(x) == 2:
            return (x[0], x[1], x[1])
        return x
    return (x, x, x)

class PatchEmbed3D(nn.Module):
    """3D patch embedding layer"""
    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (
            (img_size[0] // patch_size[0]) * 
            (img_size[1] // patch_size[1]) * 
            (img_size[2] // patch_size[2]))
        
        # 3D convolution for patch extraction
        self.proj = nn.Conv3d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        # Validate input dimensions
        assert D % self.patch_size[0] == 0 and H % self.patch_size[1] == 0 and W % self.patch_size[2] == 0, \
               f"Input dimensions {(D,H,W)} must be divisible by patch size {self.patch_size}"
        x = self.proj(x)  # [B, embed_dim, D', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class RelativePositionBias3D(nn.Module):
    """3D relative position bias for attention"""
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size  # (depth, height, width)
        self.num_heads = num_heads
        
        # 3D relative position table
        self.num_relative_distance = (
            (2 * window_size[0] - 1) * 
            (2 * window_size[1] - 1) * 
            (2 * window_size[2] - 1)) + 3
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))
        
        # Generate 3D position indices
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(
            coords_d, coords_h, coords_w, indexing='ij'))  # [3, D, H, W]
        coords_flatten = torch.flatten(coords, 1)  # [3, D*H*W]
        
        # Calculate relative positions
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [3, D*H*W, D*H*W]
        
        # Shift to start from 0
        relative_coords[0] += window_size[0] - 1
        relative_coords[1] += window_size[1] - 1
        relative_coords[2] += window_size[2] - 1
        
        # Create scalar indices
        relative_coords[0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(0)  # [D*H*W, D*H*W]
        
        # Add special tokens (CLS)
        full_index = torch.zeros(
            (window_size[0] * window_size[1] * window_size[2] + 1,) * 2,
            dtype=relative_coords.dtype
        )
        full_index[1:, 1:] = relative_position_index
        full_index[0, 0:] = self.num_relative_distance - 3
        full_index[0:, 0] = self.num_relative_distance - 2
        full_index[0, 0] = self.num_relative_distance - 1
        
        self.register_buffer("relative_position_index", full_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2] + 1,
                self.window_size[0] * self.window_size[1] * self.window_size[2] + 1, -1)
        return relative_position_bias.permute(2, 0, 1).contiguous()

class Attention3D(nn.Module):
    """3D multi-head attention with relative position bias"""
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if window_size:
            self.window_size = to_3tuple(window_size)
            self.relative_position_bias = RelativePositionBias3D(self.window_size, num_heads)
        else:
            self.window_size = None

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add 3D relative position bias
        if self.window_size:
            attn = attn + self.relative_position_bias()
            
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block3D(nn.Module):
    """3D transformer block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., init_values=None, 
                 norm_layer=nn.LayerNorm, window_size=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention3D(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size
        )
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class VisionTransformer3D(nn.Module):
    """3D Vision Transformer"""
    def __init__(self, img_size=(128,128,128), patch_size=(16,16,16), in_chans=1, 
                 num_classes=0, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=False, use_checkpoint=False):
        super().__init__()
        self.img_size = to_3tuple(img_size)
        self.patch_size = to_3tuple(patch_size)
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        
        
        # 3D patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Relative position bias
        if use_shared_rel_pos_bias:
            D = img_size[0] // patch_size[0]
            H = img_size[1] // patch_size[1]
            W = img_size[2] // patch_size[2]
            self.rel_pos_bias = RelativePositionBias3D(
                window_size=(D, H, W),  # Utiliser les dimensions réelles
                num_heads=num_heads
            )
        else:
            self.rel_pos_bias = None
        self.use_rel_pos_bias = use_rel_pos_bias
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # 3D transformer blocks
        self.blocks = nn.ModuleList([
            Block3D(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, init_values=init_values,
                window_size=(D, H, W) if use_rel_pos_bias else None 
            )
            for i in range(depth)])
        
        # Initialization
        if use_abs_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp[-2].weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_2d_weights(self, state_dict):
        """Adapt 2D pretrained weights to 3D"""
        own_state = self.state_dict()
        
        for name, param in state_dict.items():
            if name not in own_state:
                continue
                
            # Handle 2D to 3D conversion for patch embedding
            if 'proj.weight' in name and param.ndim == 4:
                # Inflate 2D conv weights to 3D by repeating along depth
                param = param.unsqueeze(2).repeat(1, 1, self.patch_size[0], 1, 1)
                param = param / self.patch_size[0]  # Normalize
                
            # Skip incompatible parameters
            if own_state[name].shape != param.shape:
                if 'pos_embed' not in name and 'relative_position' not in name:
                    print(f"⚠️ Skipping {name} due to shape mismatch")
                continue
                
            own_state[name].copy_(param)
        
        # Initialize 3D-specific components
        self._init_3d_pos_embed()
        print("✅ Successfully loaded 2D pretrained weights")

    def _init_3d_pos_embed(self):
        """Initialize 3D position embeddings from 2D"""
        if self.pos_embed is None:
            return
            
        # Get 2D positional embeddings (excluding CLS token)
        pos_embed_2d = self.pos_embed[:, 1:]
        h_2d = int(math.sqrt(pos_embed_2d.shape[1]))
        
        # Reshape to 2D grid and interpolate to 3D
        pos_embed_2d = pos_embed_2d.reshape(1, h_2d, h_2d, -1).permute(0, 3, 1, 2)
        pos_embed_3d = F.interpolate(
            pos_embed_2d,
            size=to_3tuple(int(h_2d)),
            mode='bicubic',
            align_corners=False
        ).permute(0, 2, 3, 4, 1).reshape(1, -1, pos_embed_2d.shape[1])
        
        # Combine with CLS token
        self.pos_embed.data = torch.cat([self.pos_embed[:, :1], pos_embed_3d], dim=1)
    
    def _forward_features(self, x):
        B, C, D, H, W = x.shape
        expected_patches = (D//self.patch_size[0]) * (H//self.patch_size[1]) * (W//self.patch_size[2])
        assert self.patch_embed.num_patches == expected_patches, "Patch calculation mismatch"
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.pos_embed is not None:
            x = x + self.pos_embed
            
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            
        return x
    def forward_features(self, x):
        if self.use_checkpoint:
            x = x.requires_grad_(True)  # Force les gradients
            x = checkpoint(self._forward_features, x)
        else:
            x = self._forward_features(x)
        return x



    def forward(self, x):
        x = self.forward_features(x)
        return x

def create_eva_vit_g(img_size=(128,128,128), patch_size=(16,16,16), in_chans=1, 
                     drop_path_rate=0.4, use_checkpoint=True, pretrained=True, 
                     precision="fp16"):
    """Create a 3D EVA-ViT model with optional pretrained weights"""
    model = VisionTransformer3D(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=0,
        embed_dim=1408,
        depth=39,
        num_heads=16,
        mlp_ratio=4.3637,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint=use_checkpoint,
        use_rel_pos_bias=True,
        use_shared_rel_pos_bias=True
    )
    
    if pretrained:
        print("Loading pretrained weights from EVA-ViT-g...")
        url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
        
        try:
            # Download with progress bar
            from lavis.common.dist_utils import download_cached_file
            cached_file = download_cached_file(url, check_hash=False, progress=True)
            
            # Load state dict
            state_dict = torch.load(cached_file, map_location="cpu")
            
            # Filter compatible weights
            new_state_dict = {}
            for k, v in state_dict.items():
                if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                    new_state_dict[k] = v
                else:
                    model_shape = model.state_dict()[k].shape if k in model.state_dict() else "Not found"
                    print(f"⚠️ Ignored incompatible key: {k} - {v.shape} vs {model_shape}")
            
            # Load filtered weights
            model.load_state_dict(new_state_dict, strict=False)
            print("✅ Successfully loaded compatible pretrained weights")
            
        except Exception as e:
            print(f"❌ Failed to load pretrained weights: {str(e)}")
            print("⚠️ Continuing with random initialization")
    
    # Handle precision
    if precision == "fp16":
        model.half()
    else:
        model.float()
    
    return model