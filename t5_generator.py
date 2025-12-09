import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import T5TokenizerFast
from lavis.models.blip2_models.blip2 import Blip2Base
from eva_vit_3d import create_eva_vit_g
from torch.nn import functional as F
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from enhanced_multimodal_projection import EnhancedMultimodalProjection

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class MedBLIPReportGeneratorT5(Blip2Base):
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=(128,128,128),
        patch_size=(16,16,16),
        drop_path_rate=0.4,
        use_grad_checkpoint=True,
        vit_precision="fp32",
        freeze_vit=True,
        num_query_token=128,
        t5_model="google/flan-t5-base",
        max_txt_len=512,
        embed_dim=768,
    ):
        super().__init__()

        # 1. Initialisation des composants visuels
        self.visual_encoder = create_eva_vit_g(
            img_size=img_size,
            patch_size=patch_size,
            pretrained=True,
            precision=vit_precision,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_grad_checkpoint
        )
        
        if not hasattr(self.visual_encoder, 'num_features'):
            self.visual_encoder.num_features = self.visual_encoder.embed_dim

        self.ln_vision = LayerNorm(self.visual_encoder.num_features)

        # 2. Initialisation Q-Former
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None

        # 3. Initialisation T5
        self.tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model,
            config=t5_config,
            torch_dtype=torch.float32
        )

        # 4. Initialisation de la couche de projection
        self.proj = EnhancedMultimodalProjection(
            qformer_dim=768,
            t5_dim=768,  
            hidden_dim=1024,
            num_attention_heads=8,
            num_fusion_layers=2,
            dropout=0.1,
            use_modality_weighting=True
        )

        self.max_txt_len = max_txt_len

        
        if freeze_vit:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            # Dégeler les derniers blocs
            for block in self.visual_encoder.blocks[-4:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Configurer les gradients pour les autres composants
        self.query_tokens.requires_grad = True
        
        # Configurer les gradients T5
        for i, layer in enumerate(self.t5_model.encoder.block):
            layer.requires_grad_(i >= 6)  # Unfreeze last 6/12 layers

        for i, layer in enumerate(self.t5_model.decoder.block):
            layer.requires_grad_(i >= 3)  # Unfreeze last 3/12 layers
        # Vérification finale
        self._verify_initialization()

    def _verify_initialization(self):
        """
        Vérifie que tous les composants sont correctement initialisés
        Compatible avec l'ancienne et la nouvelle architecture de projection
        """
        print("\n" + "="*50)
        print("🔍 VÉRIFICATION DE L'INITIALISATION")
        print("="*50)
        
        # 1. VISUAL ENCODER
        try:
            visual_params = list(self.visual_encoder.parameters())
            if visual_params:
                visual_trainable = visual_params[0].requires_grad
                visual_count = sum(p.numel() for p in visual_params)
                print(f"✅ Visual encoder: {'entraînable' if visual_trainable else 'gelé'} ({visual_count:,} paramètres)")
            else:
                print("⚠️ Visual encoder: aucun paramètre trouvé")
        except Exception as e:
            print(f"❌ Visual encoder: erreur ({e})")
        
        # 2. PROJECTION LAYER - Compatible ancienne et nouvelle architecture
        try:
            proj_params = list(self.proj.parameters())
            if proj_params:
                proj_trainable = proj_params[0].requires_grad
                proj_count = sum(p.numel() for p in proj_params)
                
                # Déterminer le type d'architecture
                if hasattr(self.proj, 'input_projection'):
                    # Nouvelle architecture (EnhancedMultimodalProjection)
                    proj_type = "Enhanced Multimodal"
                    print(f"✅ Projection layer ({proj_type}): {'entraînable' if proj_trainable else 'gelé'} ({proj_count:,} paramètres)")
                    
                    # Informations détaillées
                    if hasattr(self.proj, 'qformer_dim'):
                        print(f"   📊 Dimensions: {self.proj.qformer_dim} → {self.proj.hidden_dim} → {self.proj.t5_dim}")
                        print(f"   🧠 Attention heads: {self.proj.num_attention_heads}")
                else:
                    # Ancienne architecture (nn.Sequential)
                    proj_type = "Sequential"
                    print(f"✅ Projection layer ({proj_type}): {'entraînable' if proj_trainable else 'gelé'} ({proj_count:,} paramètres)")
            else:
                print("⚠️ Projection layer: aucun paramètre trouvé")
        except Exception as e:
            print(f"❌ Projection layer: erreur ({e})")
        
        # 3. T5 MODEL
        try:
            t5_params = list(self.t5_model.parameters())
            if t5_params:
                t5_trainable = t5_params[0].requires_grad
                t5_count = sum(p.numel() for p in t5_params)
                print(f"✅ T5 model: {'entraînable' if t5_trainable else 'gelé'} ({t5_count:,} paramètres)")
            else:
                print("⚠️ T5 model: aucun paramètre trouvé")
        except Exception as e:
            print(f"❌ T5 model: erreur ({e})")
        
        print("="*50)
        print("✅ VÉRIFICATION TERMINÉE")
        print("="*50)

    def forward(self, samples, generate=False):
        # 1. Préparation et nettoyage des entrées
        image_mri = torch.nan_to_num(samples["mri"].to(self.device).float(), nan=0.0, posinf=5.0, neginf=-5.0)
        image_pet = torch.nan_to_num(samples["pet"].to(self.device).float(), nan=0.0, posinf=5.0, neginf=-5.0)
        clinical_text = samples["clinical_text"]
        reports = samples.get("reports", [""])  # reports peut être absent en génération

        if torch.isnan(image_mri).any() or torch.isinf(image_mri).any():
            print("Avertissement: MRI contient des valeurs invalides après nettoyage")
        if torch.isnan(image_pet).any() or torch.isinf(image_pet).any():
            print("Avertissement: PET contient des valeurs invalides après nettoyage")

        # 2. Encodage visuel
        with torch.cuda.amp.autocast(enabled=False):
            try:
                mri_feats = torch.clamp(self.ln_vision(self.visual_encoder(image_mri)), -5, 5)
                pet_feats = torch.clamp(self.ln_vision(self.visual_encoder(image_pet)), -5, 5)
            except Exception as e:
                print(f"Erreur critique dans l'encodeur visuel: {str(e)}")
                torch.cuda.empty_cache()
                return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

        # 3. Q-Former
        try:
            image_embeds = torch.cat([mri_feats, pet_feats], dim=1)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=torch.clamp(image_embeds, -10, 10),
                return_dict=True
            )
            img_feats, _ = self.proj(query_output.last_hidden_state)
        except Exception as e:
            print(f"Erreur dans Q-Former: {str(e)}")
            torch.cuda.empty_cache()
            return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

        # 4. Texte : tokenization clinique et rapport
        try:
            clinical_tokens = self.tokenizer(
                clinical_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len // 2,
                return_tensors="pt"
            ).to(self.device)

            if not generate:
                report_tokens = self.tokenizer(
                    reports,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len // 2,
                    return_tensors="pt"
                ).to(self.device)

            txt_embeds = self.t5_model.encoder.embed_tokens(clinical_tokens.input_ids)
            inputs_embeds = torch.cat([img_feats, txt_embeds], dim=1)

            attention_mask = torch.cat([
                torch.ones(img_feats.shape[:2], dtype=torch.long, device=self.device),
                clinical_tokens.attention_mask
            ], dim=1)

            if generate:
                generated_ids = self.t5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=True,      
                    temperature=0.7,    
                    max_new_tokens= 400,  
                    min_length= 200,
                    top_p=0.9,      
                    repetition_penalty=1.5,             
                    top_k=50,
                    length_penalty=1.5,  
                    early_stopping=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                predictions = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                return {"predictions": predictions}
            else:
                labels = report_tokens.input_ids.masked_fill(
                    report_tokens.input_ids == self.tokenizer.pad_token_id, -100
                )
                outputs = self.t5_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )
                if torch.isnan(outputs.loss):
                    print("Avertissement: Loss NaN. Utilisation de fallback...")
                    outputs.loss = torch.tensor(1.0, device=self.device, requires_grad=True)
                return {"loss": outputs.loss}

        except Exception as e:
            print(f"Erreur dans le traitement T5: {str(e)}")
            torch.cuda.empty_cache()
            return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

    @torch.no_grad()
    def generate_report(self, samples, max_new_tokens=300, temperature=0.7):
        image_mri = samples["mri"].to(self.device)
        image_pet = samples["pet"].to(self.device)
        clinical_text = samples["clinical_text"]

        with autocast(enabled=self.t5_model.dtype == torch.float16):
            mri_feats = self.ln_vision(self.visual_encoder(image_mri))
            pet_feats = self.ln_vision(self.visual_encoder(image_pet))

            image_embeds = torch.cat([mri_feats, pet_feats], dim=1)

            # Q-Former processing
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                return_dict=True
            )
            img_feats , _ = self.proj(query_output.last_hidden_state)

        # Tokenize clinical context
        clinical_tokens = self.tokenizer(
            clinical_text,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt"
        ).to(self.device)

        # Combine image features and clinical text
        inputs_embeds = self.t5_model.encoder.embed_tokens(clinical_tokens.input_ids)
        inputs_embeds = torch.cat([img_feats, inputs_embeds], dim=1)
        attention_mask = torch.cat([
            torch.ones(img_feats.shape[:2], dtype=torch.long, device=self.device),
            clinical_tokens.attention_mask
        ], dim=1)

        # Generate report
        generated_ids = self.t5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        reports = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        formatted_reports = [self._format_report(r) for r in reports]
        return formatted_reports

    def _format_report(self, raw_text: str):
        return raw_text.strip()

    @property
    def device(self):
        return next(self.parameters()).device