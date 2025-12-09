
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class EnhancedMultimodalProjection(nn.Module):
    
    def __init__(self, 
                 qformer_dim: int = 768,
                 t5_dim: int = 512,
                 hidden_dim: int = 1024,
                 num_attention_heads: int = 8,
                 num_fusion_layers: int = 2,
                 dropout: float = 0.2,
                 use_modality_weighting: bool = True):
        super().__init__()
        
        self.qformer_dim = qformer_dim
        self.t5_dim = t5_dim
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.use_modality_weighting = use_modality_weighting
        
        # Vérification des dimensions
        assert hidden_dim % num_attention_heads == 0, \
            f"hidden_dim ({hidden_dim}) doit être divisible par num_attention_heads ({num_attention_heads})"
        
        print(f"🏗️ EnhancedMultimodalProjection:")
        print(f"   Input: {qformer_dim} → Hidden: {hidden_dim} → Output: {t5_dim}")
        print(f"   Attention heads: {num_attention_heads}, Fusion layers: {num_fusion_layers}")
        
        # 1. PROJECTION D'ENTRÉE
        self.input_projection = nn.Sequential(
            nn.Linear(qformer_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. ATTENTION MULTIMODALE
        self.multimodal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. COUCHES DE FUSION PROFONDES
        self.fusion_layers = nn.ModuleList()
        for i in range(num_fusion_layers):
            layer = nn.ModuleDict({
                'linear1': nn.Linear(hidden_dim, hidden_dim * 2),
                'linear2': nn.Linear(hidden_dim * 2, hidden_dim),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
                'dropout': nn.Dropout(dropout)
            })
            self.fusion_layers.append(layer)
        
        # 4. PROJECTION DE SORTIE
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, t5_dim)
        )
        
        # 5. CONNEXION RÉSIDUELLE ADAPTATIVE
        if qformer_dim == t5_dim:
            self.residual_connection = nn.Identity()
        else:
            self.residual_connection = nn.Sequential(
                nn.Linear(qformer_dim, t5_dim),
                nn.LayerNorm(t5_dim)
            )
        
        # 6. PONDÉRATION DES MODALITÉS
        if use_modality_weighting:
            self.modality_gate = nn.Sequential(
                nn.Linear(t5_dim, t5_dim // 4),
                nn.GELU(),
                nn.Linear(t5_dim // 4, 1),
                nn.Sigmoid()
            )
        
        # 7. NORMALISATION FINALE
        self.final_norm = nn.LayerNorm(t5_dim)
        
        # Initialisation des poids
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation spécialisée pour la fusion multimodale"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'output_projection' in name:
                    # Initialisation plus conservative pour la sortie
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_len, _ = x.shape
        
        # Sauvegarder l'entrée pour la connexion résiduelle
        residual_input = x
        
        # 1. PROJECTION D'ENTRÉE
        x = self.input_projection(x)  # [B, S, hidden_dim]
        
        # 2. ATTENTION MULTIMODALE
        # Auto-attention pour capturer les relations inter-tokens
        attn_output, attention_weights = self.multimodal_attention(
            query=x,
            key=x, 
            value=x,
            key_padding_mask=~attention_mask if attention_mask is not None else None,
            need_weights=return_attention
        )
        
        # Connexion résiduelle après attention
        x = x + attn_output
        
        # 3. COUCHES DE FUSION PROFONDES
        for i, fusion_layer in enumerate(self.fusion_layers):
            # Sauvegarder pour connexion résiduelle
            layer_residual = x
            
            # Première transformation
            x = fusion_layer['norm1'](x)
            x = fusion_layer['linear1'](x)
            x = F.gelu(x)
            x = fusion_layer['dropout'](x)
            
            # Deuxième transformation
            x = fusion_layer['linear2'](x)
            x = fusion_layer['dropout'](x)
            
            # Connexion résiduelle
            x = layer_residual + x
            x = fusion_layer['norm2'](x)
        
        # 4. PROJECTION VERS L'ESPACE T5
        x = self.output_projection(x)  # [B, S, t5_dim]
        
        # 5. CONNEXION RÉSIDUELLE PRINCIPALE
        residual_projected = self.residual_connection(residual_input)
        x = x + residual_projected
        
        # 6. PONDÉRATION DES MODALITÉS (optionnel)
        if self.use_modality_weighting:
            modality_weights = self.modality_gate(x)
            x = x * modality_weights
        
        # 7. NORMALISATION FINALE
        x = self.final_norm(x)
        
        if return_attention:
            return x, attention_weights
        else:
            return x, None
    
    def get_attention_maps(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        
        with torch.no_grad():
            _, attention_weights = self.forward(x, attention_mask, return_attention=True)
        return attention_weights
    
    def compute_fusion_statistics(self, x: torch.Tensor):
        
        with torch.no_grad():
            # Projection d'entrée
            x_proj = self.input_projection(x)
            
            # Statistiques
            stats = {
                'input_mean': x.mean().item(),
                'input_std': x.std().item(),
                'projected_mean': x_proj.mean().item(),
                'projected_std': x_proj.std().item(),
                'activation_sparsity': (x_proj == 0).float().mean().item()
            }
            
            return stats

# Fonction utilitaire pour créer la couche
def create_enhanced_projection(qformer_config, t5_config, **kwargs):
   
    return EnhancedMultimodalProjection(
        qformer_dim=qformer_config.hidden_size,
        t5_dim=t5_config.hidden_size,
        **kwargs
    )

