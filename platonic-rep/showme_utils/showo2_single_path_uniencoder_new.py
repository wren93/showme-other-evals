# pyre-unsafe
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from showme_utils.omni_attention import (
    omni_attn_mask_flexattention_interleaved,
    omni_attn_mask_naive,
)

logger: logging.Logger = logging.getLogger(__name__)


class ShowO2SinglePathModel_uni(nn.Module):
    def __init__(
        self,
        # uniencoder configuration
        uniencoder_cfg_path: Optional[str] = None,
        uniencoder_ckpt_path: Optional[str] = None,
        uniencoder_feature_layer: Union[str, int] = "post_norm",
        uniencoder_init_vae_decoder: bool = False,
        uniencoder_attention_implementation: str = "sdpa",
        # Showo2 configuration
        llm_model_path: str = "",
        load_stage1_model: Optional[str] = None,
        frozen_params: Optional[List[str]] = None,
        hidden_size: int = 1536,
        image_latent_dim: int = 16,
        image_latent_height: int = 27,
        image_latent_width: int = 27,
        hq_image_latent_height: int = 64,
        hq_image_latent_width: int = 64,
        mixed_modal_latent_height: int = 27,
        mixed_modal_latent_width: int = 27,
        patch_size: int = 2,
        clip_latent_dim: int = 1152,
        add_time_embeds: bool = True,
        add_aspect_ratio_embeds: bool = True,
        mrope_type: str = "none",
        reshape_frame_to_batch_dim: bool = False,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        attention_backend: str = "sdpa",
        # Training configuration
        ntp_coeff: float = 1.0,
        flow_coeff: float = 1.0,
        und_max_t0: float = 1.0,
        use_disp: bool = False,
        gradient_checkpointing: bool = False,
        gradient_checkpointing_kwargs: Optional[DictConfig] = None,
        # Transport configuration
        path_type: str = "Linear",
        prediction: str = "velocity",
        loss_weight: Optional[str] = None,
        train_eps: Optional[float] = 1e-5,
        sample_eps: Optional[float] = 1e-3,
        snr_type: str = "uniform",
        do_shift: bool = False,
        # Sampling configuration
        sampling_method: str = "euler",
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
        time_shifting_factor: float = 3.0,
        # Data configuration
        dtype: str = "bf16",
        flow_head_num: int = 10,
    ):
        super().__init__()

        # uniencoder configuration
        self.uniencoder_cfg_path = uniencoder_cfg_path
        self.uniencoder_ckpt_path = uniencoder_ckpt_path
        self.uniencoder_feature_layer = uniencoder_feature_layer
        self.uniencoder_init_vae_decoder = uniencoder_init_vae_decoder
        self.uniencoder_attention_implementation = uniencoder_attention_implementation

        # Showo2 configuration
        self.llm_model_path = llm_model_path
        self.frozen_params = frozen_params
        self.hidden_size = hidden_size
        self.image_latent_dim = image_latent_dim
        self.image_latent_height = image_latent_height
        self.image_latent_width = image_latent_width
        self.hq_image_latent_height = hq_image_latent_height
        self.hq_image_latent_width = hq_image_latent_width
        self.mixed_modal_latent_height = mixed_modal_latent_height
        self.mixed_modal_latent_width = mixed_modal_latent_width
        self.patch_size = patch_size
        self.clip_latent_dim = clip_latent_dim
        self.add_time_embeds = add_time_embeds
        self.add_aspect_ratio_embeds = add_aspect_ratio_embeds
        self.load_stage1_model = load_stage1_model
        self.flow_head_num = flow_head_num
        self.mrope_type = mrope_type
        self.reshape_frame_to_batch_dim = reshape_frame_to_batch_dim
        self.attention_backend = attention_backend

        # Training coefficients
        self.ntp_coeff = ntp_coeff
        self.flow_coeff = flow_coeff
        self.und_max_t0 = und_max_t0
        self.use_disp = use_disp
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_kwargs = OmegaConf.to_container(
            gradient_checkpointing_kwargs
        )

        # Transport configuration
        self.path_type = path_type
        self.prediction = prediction
        self.loss_weight = loss_weight
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.snr_type = snr_type
        self.do_shift = do_shift

        # Sampling configuration
        self.sampling_method = sampling_method
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.atol = atol
        self.rtol = rtol
        self.reverse = reverse
        self.time_shifting_factor = time_shifting_factor

        # Device and dtype
        self.dtype = dtype

        # Build models
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.build_models()

    def build_models(self):
        """Initialize all model components"""
        from showme_utils.misc import get_text_tokenizer
        from showme_utils.transport.define import create_transport, Sampler

        # Import here to avoid circular imports
        if self.mrope_type == "dit_3drope_mm":
            from showme_utils.modeling_showo2_qwen2_5_single_path_uniencoder_3drope_dit_mm import (
                Showo2Qwen2_5SinglePath_uniencoder,
            )

        # Initialize text tokenizer
        llm_local_path = self.llm_model_path
        self.text_tokenizer, self.showo_token_ids = get_text_tokenizer(
            llm_local_path,
            add_showo_tokens=True,
            return_showo_token_ids=True,
        )
        self.llm_vocab_size = len(self.text_tokenizer)

        # Initialize Show-o model with single path architecture
        model_config = {
            "uniencoder_cfg_path": self.uniencoder_cfg_path,
            "uniencoder_ckpt_path": self.uniencoder_ckpt_path,
            "uniencoder_feature_layer": self.uniencoder_feature_layer,
            "uniencoder_init_vae_decoder": self.uniencoder_init_vae_decoder,
            "uniencoder_attention_implementation": self.uniencoder_attention_implementation,
            "llm_vocab_size": self.llm_vocab_size,
            "llm_model_path": llm_local_path,
            "load_from_showo": False,
            "image_latent_dim": self.image_latent_dim,
            "image_latent_height": self.image_latent_height,
            "image_latent_width": self.image_latent_width,
            "video_latent_height": self.image_latent_height,  # Using image_latent_height as default
            "video_latent_width": self.image_latent_width,  # Using image_latent_width as default
            "reshape_frame_to_batch_dim": self.reshape_frame_to_batch_dim,
            "hidden_size": self.hidden_size,
            "patch_size": self.patch_size,
            "clip_latent_dim": self.clip_latent_dim,
            "add_time_embeds": self.add_time_embeds,
            "add_aspect_ratio_embeds": self.add_aspect_ratio_embeds,
            "num_diffusion_layers": self.flow_head_num,
            "gradient_checkpointing": self.gradient_checkpointing,
            "gradient_checkpointing_kwargs": self.gradient_checkpointing_kwargs,
            "use_disp": self.use_disp,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
        }

        self.showo_model = Showo2Qwen2_5SinglePath_uniencoder(**model_config)
        if self.load_stage1_model is not None and self.load_stage1_model != "no":
            stage1_model_path = self.load_stage1_model
            self.showo_model.load_state_dict(
                torch.load(stage1_model_path, map_location="cpu")
            )

        self._freeze_params(self.showo_model, self.frozen_params)

        # Initialize transport for flow matching
        self.transport = create_transport(
            path_type=self.path_type,
            prediction=self.prediction,
            loss_weight=self.loss_weight,
            train_eps=self.train_eps,
            sample_eps=self.sample_eps,
            snr_type=self.snr_type,
            do_shift=self.do_shift,
        )
        logger.info("loaded all pretrained model!")
        self.sampler = Sampler(self.transport)

    def _freeze_params(self, model, frozen_params=None):
        if frozen_params is not None:
            for n, p in model.named_parameters():
                for name in frozen_params:
                    if name in n:
                        p.requires_grad = False

    @torch.no_grad()
    def prepare_latents_and_labels(
        self,
        pixel_values: torch.Tensor,
        data_type: List[str],
        image_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare image latents and labels for training - unified interface"""

        if len(pixel_values.shape) == 4:
            pixel_values = pixel_values.unsqueeze(2)

        image_latents = self.showo_model.vision_model.vae_encode(
            pixel_values, deterministic=False
        )

        # Prepare timesteps, noise, and targets
        t_list, xt_list, ut_list, masks = [], [], [], []

        for i, tp in enumerate(data_type):
            # Sample timestep and noise
            t, x0, x1 = self.transport.sample(
                image_latents[i][None],
                (
                    self.und_max_t0
                    if tp in ["mmu", "mmu_vid", "mmu_interleaved", "mmu_text"]
                    else None
                ),
            )
            # Get noisy latents and velocity targets
            t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)

            t_list.append(t)
            xt_list.append(xt)
            ut_list.append(ut)

            # Handle masks for understanding tasks
            if (
                tp in ["mmu", "mmu_vid", "mmu_interleaved", "mmu_text"]
                and self.und_max_t0 == 1.0
            ):
                if i < image_masks.shape[0]:
                    masks.append(image_masks[i][None] * 0.0)
            else:
                masks.append(image_masks[i][None])

        t = torch.stack(t_list, dim=0).squeeze(-1)
        xt = torch.cat(xt_list, dim=0)
        ut = torch.cat(ut_list, dim=0)
        masks = torch.cat(masks, dim=0) if masks else image_masks

        # Always return both clean and noisy latents for consistency
        return xt, t, ut, masks, image_latents

    def create_attention_mask(
        self,
        batch_size: int,
        seq_length: int,
        modality_positions: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Create omni attention mask"""
        if self.attention_backend == "sdpa":
            attention_mask = omni_attn_mask_naive(
                batch_size, seq_length, modality_positions, device
            ).to(dtype)
            return attention_mask, None
        elif self.attention_backend == "flexattention":
            attention_mask = omni_attn_mask_flexattention_interleaved(
                modality_positions,
                seq_length,
                self.showo_model.showo.config.num_attention_heads,
                device=device,
            )
            attention_mask_diffhead = omni_attn_mask_flexattention_interleaved(
                modality_positions,
                seq_length,
                self.showo_model.diffusion_head_config.num_attention_heads,
                device=device,
            )
            return attention_mask, attention_mask_diffhead
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass for training"""
        # Extract batch data
        weight_type = torch.bfloat16 if self.dtype == "bf16" else torch.float32
        text_tokens = batch["text_tokens"]
        text_labels = batch["text_labels"]
        pixel_values = batch["images"]
        text_masks = batch["text_masks"]
        image_masks = batch["image_masks"]
        modality_positions = batch["modality_positions"]
        data_type = batch["data_type"]

        # Handle interleaved data
        if data_type[0] == "mmu_interleaved":
            b, n = pixel_values.shape[:2]
            pixel_values = rearrange(pixel_values, "b n c h w -> (b n) c h w")
            data_type = data_type * n

        if data_type[0] != "mmu_text":
            # Prepare image latents and labels
            image_latents, t, image_labels, image_masks, image_latents_clean = (
                self.prepare_latents_and_labels(pixel_values, data_type, image_masks)
            )
        else:
            image_latents = None
            t = torch.tensor([0.0] * text_tokens.shape[0], device=text_tokens.device)
            image_labels = None
            image_masks = None

        # Create attention mask
        block_mask, block_mask_diffhead = self.create_attention_mask(
            text_tokens.size(0),
            text_tokens.size(1),
            modality_positions,
            text_tokens.device,
            weight_type,
        )

        # Forward pass through the model
        logits, loss_ntp, loss_flow, loss_disp = self.showo_model(
            text_tokens=text_tokens,
            image_latents=image_latents,
            t=t.to(weight_type),
            attention_mask=block_mask,
            diffhead_attention_mask=block_mask_diffhead,
            text_masks=text_masks,
            image_masks=image_masks,
            text_labels=text_labels,
            image_labels=image_labels,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=text_tokens.size(1),
            device=text_tokens.device,
        )

        # Compute total loss
        total_loss = self.flow_coeff * loss_flow + self.ntp_coeff * loss_ntp
        if self.use_disp:
            total_loss += 0.25 * loss_disp

        outputs = {
            "loss": total_loss,
            "loss_ntp": loss_ntp,
            "loss_flow": loss_flow,
            "loss_disp": loss_disp,
            "logits": logits,
            "recons_images": None,
        }

        return outputs
