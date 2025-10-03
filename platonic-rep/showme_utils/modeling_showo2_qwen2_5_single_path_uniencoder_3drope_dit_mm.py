# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from omegaconf import OmegaConf

from showme_utils.misc import next_token_prediction, velocity_prediction
from showme_utils.modules_new_mm import (
    DiffusionHeadConfig,
    FinalLayer,
    ModulatedAttentionBlock,
    RMSNorm,
    TimestepEmbedder,
)
from showme_utils.omni_attention import step_block_mask_from_old

from showme_utils.qwen2 import Qwen2ForCausalLM
from showme_utils.siglip_vae.configuration_siglip_vae import SiglipConfig
from showme_utils.siglip_vae.modelling_siglip_vae import SiglipModelWithVAE
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig

logger: logging.Logger = logging.getLogger(__name__)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)  # pyre-ignore
    out = torch.einsum("n,d->nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "n d (i j) -> n d i j", i=2, j=2)
    return out.float()


def build_rope(latent_shape, patch_size, attention_head_dim):
    dim_t = attention_head_dim // 4
    dim_h = attention_head_dim // 8 * 3
    dim_w = attention_head_dim // 8 * 3
    assert (
        dim_t + dim_h + dim_w == attention_head_dim
    ), f"{dim_t + dim_h + dim_w} != {attention_head_dim}"

    latent_t = latent_shape[0]
    latent_h = latent_shape[1] // patch_size
    latent_w = latent_shape[2] // patch_size
    visual_ids = torch.zeros(latent_t, latent_h, latent_w, 3)
    visual_ids[..., 0] = visual_ids[..., 0] + torch.arange(latent_t)[:, None, None]
    visual_ids[..., 1] = visual_ids[..., 1] + torch.arange(latent_h)[None, :, None]
    visual_ids[..., 2] = visual_ids[..., 2] + torch.arange(latent_w)[None, None, :]
    visual_ids = rearrange(visual_ids, "t h w c -> (t h w) c")

    rope_3d = torch.cat(
        [
            rope(visual_ids[..., 0], dim_t, 10_000),
            rope(visual_ids[..., 1], dim_h, 10_000),
            rope(visual_ids[..., 2], dim_w, 10_000),
        ],
        dim=-3,
    )
    return rope_3d


class Showo2Qwen2_5SinglePath_uniencoder(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        uniencoder_cfg_path=None,
        uniencoder_ckpt_path=None,
        uniencoder_feature_layer="post_norm",
        uniencoder_init_vae_decoder=False,
        uniencoder_attention_implementation="sdpa",
        llm_vocab_size=None,
        llm_model_path="",
        load_from_showo=False,
        image_latent_dim=16,
        image_latent_height=16,
        image_latent_width=16,
        video_latent_height=16,
        video_latent_width=16,
        reshape_frame_to_batch_dim=False,
        num_attention_heads=24,
        num_key_value_heads=8,
        patch_size=2,
        hidden_size=2048,
        clip_latent_dim=1152,
        num_diffusion_layers=10,
        add_aspect_ratio_embeds=True,
        add_time_embeds=True,
        use_disp=False,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        **kwargs,
    ):
        super().__init__()
        self.use_disp = use_disp

        llm_config = AutoConfig.from_pretrained(llm_model_path)
        if load_from_showo:
            self.showo = Qwen2ForCausalLM(llm_config)
        else:
            self.showo = Qwen2ForCausalLM.from_pretrained(
                llm_model_path, attn_implementation="sdpa"
            )
        self.showo.resize_token_embeddings(llm_vocab_size)

        # Single path: VAE -> SigLIP -> Transformer
        local_model_cfg_file = uniencoder_cfg_path
        model_cfg = OmegaConf.load(local_model_cfg_file)
        vision_encoder_path = model_cfg.model_name_or_path
        if vision_encoder_path.startswith("manifold://"):
            vision_encoder_path = vision_encoder_path.replace(
                "manifold://", "/home/wren93/"
            )
        sigvae_config = SiglipConfig.from_pretrained(vision_encoder_path)
        if model_cfg.vision_config is not None:
            sigvae_config.vision_config.update(
                OmegaConf.to_container(model_cfg.vision_config)
            )
        if model_cfg.text_config is not None:
            sigvae_config.text_config.update(
                OmegaConf.to_container(model_cfg.text_config)
            )
        sigvae_config._attn_implementation = uniencoder_attention_implementation
        sigvae_config.vision_config.vae_init_decoder = uniencoder_init_vae_decoder
        sigvae_config.vision_config.vae_use_projection = False
        sigvae_config.vision_config.vision_use_head = False

        model = SiglipModelWithVAE.from_pretrained(
            vision_encoder_path,
            config=sigvae_config,
            ignore_mismatched_sizes=True,
        )
        model.load_vae()

        self.uniencoder_ckpt_path = uniencoder_ckpt_path
        if uniencoder_ckpt_path is not None:
            local_state_dict_path = uniencoder_ckpt_path
            state_dict = torch.load(local_state_dict_path)
            m, u = model.load_state_dict(state_dict, strict=False)
            logger.info(f"missing keys: {m}")
            logger.info(f"unexpected keys: {u}")

        uniencoder_num_layers = model.config.vision_config.num_hidden_layers
        self.uniencoder_feature_layer = uniencoder_feature_layer
        if self.uniencoder_feature_layer != "post_norm":
            model.vision_model.post_layernorm = nn.Identity()
            if self.uniencoder_feature_layer < 0:
                feature_layer = (
                    uniencoder_num_layers + self.uniencoder_feature_layer + 1
                )
            else:
                feature_layer = self.uniencoder_feature_layer + 1
            model.vision_model.encoder.layers = model.vision_model.encoder.layers[
                :feature_layer
            ]

        self.register_buffer(
            "image_position_ids",
            torch.arange(image_latent_height * image_latent_width).expand((1, -1)),
            persistent=False,
        )

        self.siglip_proj = nn.Sequential(
            RMSNorm(clip_latent_dim),
            nn.Linear(clip_latent_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Diffusion head for generation
        self.diffusion_head_config = DiffusionHeadConfig(
            hidden_size=self.showo.config.hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=self.showo.config.intermediate_size,
            max_position_embeddings=self.showo.config.max_position_embeddings,
        )
        self.time_embed = TimestepEmbedder(self.diffusion_head_config.hidden_size)
        if add_aspect_ratio_embeds:
            self.aspect_ratio_embed = TimestepEmbedder(
                self.diffusion_head_config.hidden_size
            )
        if hidden_size != self.diffusion_head_config.hidden_size:
            self.diff_proj = nn.Sequential(
                nn.Linear(hidden_size, self.diffusion_head_config.hidden_size),
                nn.GELU(),
                nn.Linear(
                    self.diffusion_head_config.hidden_size,
                    self.diffusion_head_config.hidden_size,
                ),
            )
            self.time_embed_proj = nn.Linear(
                self.diffusion_head_config.hidden_size, hidden_size
            )
            if add_aspect_ratio_embeds:
                self.ar_embed_proj = nn.Linear(
                    self.diffusion_head_config.hidden_size, hidden_size
                )
        self.diffusion_head_a = nn.ModuleList(
            [
                ModulatedAttentionBlock(self.diffusion_head_config, layer_idx)
                for layer_idx in range(num_diffusion_layers)
            ]
        )
        self.diffusion_head_b = FinalLayer(
            self.diffusion_head_config.hidden_size, patch_size, image_latent_dim
        )

        self.gradient_checkpointing = False
        if gradient_checkpointing:
            self.gradient_checkpointing = True
            self._gradient_checkpointing_func = functools.partial(
                checkpoint, **gradient_checkpointing_kwargs
            )
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
            self.showo.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

        self.vision_model = model.vision_model

        self.reset_parameters()

    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value

    def reset_parameters(self):
        # Initialize image embedders
        if self.uniencoder_ckpt_path is None:
            w1 = self.vision_model.embeddings.patch_embedding.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            nn.init.constant_(self.vision_model.embeddings.patch_embedding.bias, 0)

        # Initialize projection layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Don't reset SigLIP parameters - keep pretrained weights
        _basic_init(self.siglip_proj)
        _basic_init(self.diffusion_head_a)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out output layers
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.diffusion_head_b.linear.weight, 0)
        nn.init.constant_(self.diffusion_head_b.linear.bias, 0)

    def disp_loss(self, z):  # Dispersive Loss implementation (InfoNCE-L2 variant)
        z = z.reshape((z.shape[0], -1))  # flatten
        diff = torch.nn.functional.pdist(z).pow(2) / z.shape[1]  # pairwise distance
        diff = torch.concat(
            (diff, diff, torch.zeros(z.shape[0]).cuda())
        )  # match JAX implementation of full BxB matrix
        return torch.log(torch.exp(-diff).mean())  # calculate loss

    def unpatchify(self, x, h, w, T=0):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.config.image_latent_dim
        p = self.config.patch_size
        if T == 0:
            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            imgs = x.reshape(shape=(x.shape[0], h * p * w * p, c))
        else:
            x = x.reshape(shape=(x.shape[0], T, h, w, p, p, c))
            imgs = x.reshape(shape=(x.shape[0], T, h * p * w * p, c))
        return imgs

    def forward(
        self,
        text_tokens=None,
        image_latents=None,
        t=None,
        attention_mask=None,
        diffhead_attention_mask=None,
        text_masks=None,
        image_masks=None,
        text_labels=None,
        image_labels=None,
        modality_positions=None,
        first_frame_as_cond=False,
        only_denoise_last_image=False,
        guidance_scale=0.0,
        output_hidden_states=True,
        max_seq_len=None,
        device="cuda:0",
        label=None,
        return_input_embeds=False,
        image_grid_thw=None,
        **kwargs,
    ):
        # multimodal understanding and generation
        input_embeds = self.showo.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype

        image_embeds = None
        if image_latents is not None:
            b, c, T, h, w = image_latents.shape
            rope_3d = build_rope(
                latent_shape=[T, h, w], patch_size=1, attention_head_dim=64
            ).to(device)
            p = self.config.patch_size
            h_, w_ = h // p, w // p

            interpolate_pos_encoding = (
                not self.vision_model.embeddings.position_embedding.weight.shape[0]
                == h_ * w_
            )

            vision_model_outputs = self.vision_model(
                hidden_states_post_vae=image_latents.to(dtype),
                interpolate_pos_encoding=interpolate_pos_encoding,
                reshape_frame_to_batch_dim=self.config.reshape_frame_to_batch_dim,
            )
            image_embeds_siglip = vision_model_outputs["last_hidden_state"]

            image_embeds = self.siglip_proj(image_embeds_siglip)

        time_embeds = self.time_embed(t, dtype)
        if hasattr(self, "time_embed_proj"):
            time_embeds_proj = self.time_embed_proj(time_embeds)
        else:
            time_embeds_proj = time_embeds

        height_embeds_proj = None
        width_embeds_proj = None
        if hasattr(self, "aspect_ratio_embed"):
            latent_height = torch.tensor(h_, device=device).repeat(b)
            latent_width = torch.tensor(w_, device=device).repeat(b)
            height_embeds = self.aspect_ratio_embed(latent_height, dtype)
            width_embeds = self.aspect_ratio_embed(latent_width, dtype)
            if hasattr(self, "ar_embed_proj"):
                height_embeds_proj = self.ar_embed_proj(height_embeds)
                width_embeds_proj = self.ar_embed_proj(width_embeds)
            else:
                height_embeds_proj = height_embeds
                width_embeds_proj = width_embeds
        # Prepare image labels for training
        # Structure text and image embeddings into sequences
        new_image_labels = None
        if image_labels is not None:
            image_labels = rearrange(image_labels, "b c t h w -> b (t h w) c")
            image_labels = image_labels.reshape(shape=(b, T, h_, w_, p, p, c))
            image_labels = image_labels.reshape(shape=(b, T * h_ * w_, p * p * c))
            p = self.config.patch_size
            c = self.config.image_latent_dim
            new_image_labels = torch.zeros(
                [image_embeds.shape[0], max_seq_len, p * p * c],
                device=device,
                dtype=dtype,
            )
            image_masks = image_masks[:, :, None].repeat(1, 1, p * p * c)

        input_embeds, new_image_labels, image_masks = self._prepare_input(
            input_embeds,
            image_embeds,
            image_labels,
            image_masks,
            new_image_labels,
            modality_positions,
            height_embeds_proj,
            width_embeds_proj,
            time_embeds_proj,
        )

        # image_grid_thw = (
        #     torch.tensor([T, h_, w_]).unsqueeze(0).repeat(b, 1).unsqueeze(1)
        # ).to(device, dtype=torch.long)

        # position_ids, mrope_position_deltas = self.get_rope_index(
        #     input_ids=text_tokens,
        #     modality_positions=modality_positions,
        #     image_grid_thw=image_grid_thw,
        #     attention_mask=attention_mask,
        # )

        if return_input_embeds:
            return input_embeds

        outputs = self.showo(
            inputs_embeds=input_embeds,
            # position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        logits, last_hidden_states = outputs["logits"], outputs["hidden_states"][-1]
        position_ids = torch.arange(
            last_hidden_states.shape[1], device=last_hidden_states.device
        ).unsqueeze(0)
        # Diffusion head to predict vector fields
        if hasattr(self, "diff_proj"):
            last_hidden_states = self.diff_proj(last_hidden_states)

        if diffhead_attention_mask is None:
            diffhead_attention_mask = attention_mask
        act = []
        for layer in self.diffusion_head_a:
            if self.gradient_checkpointing and self.training:
                last_hidden_states = self._gradient_checkpointing_func(
                    layer,  # 直接传 layer，可调用
                    hidden_states=last_hidden_states,
                    adaln_input=time_embeds,
                    attention_mask=diffhead_attention_mask,
                    position_ids=position_ids,
                    rope_3d=rope_3d,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=None,
                    position_embeddings=None,
                    modality_positions=modality_positions,
                )[0]
            else:
                last_hidden_states = layer(
                    hidden_states=last_hidden_states,
                    adaln_input=time_embeds,
                    attention_mask=diffhead_attention_mask,
                    position_ids=position_ids,
                    rope_3d=rope_3d,
                    modality_positions=modality_positions,
                )[0]
            act.append(last_hidden_states)
        v_pred = self.diffusion_head_b(
            last_hidden_states, time_embeds, modality_positions
        )

        loss_disp = torch.tensor(0.0, device=device)
        if image_latents is None:
            loss_ntp = next_token_prediction(
                logits, text_labels, self.config.llm_vocab_size
            )
            loss_flow = torch.tensor(0.0, device=device)
            return logits, loss_ntp, loss_flow, loss_disp

        if text_labels is not None and image_labels is not None:
            loss_ntp = next_token_prediction(
                logits, text_labels, self.config.llm_vocab_size
            )
            loss_flow = velocity_prediction(
                v_pred, new_image_labels[: v_pred.shape[0]], image_masks
            )
            if self.use_disp:
                loss_disp = self.disp_loss(act[-1])
            return logits, loss_ntp, loss_flow, loss_disp

        else:
            # Inference mode - return velocity predictions
            v_pred_ = []
            num_imgs = 0
            for i, modality_batch in enumerate(modality_positions):
                for _, (offset, length) in enumerate(modality_batch):
                    if length == 0:
                        break
                    else:
                        v_pred_.append(v_pred[i, offset : offset + length])
                        num_imgs += 1
            v_pred_ = torch.stack(v_pred_)

            # Remove the time embedding
            if self.config.add_time_embeds and self.config.add_aspect_ratio_embeds:
                v_pred_ = v_pred_[:, 3:, :]
            elif self.config.add_time_embeds:
                v_pred_ = v_pred_[:, 1:, :]

            # Unpatchify
            v_pred_ = self.unpatchify(v_pred_, h_, w_, T=T)

            v_pred_ = v_pred_.permute(0, 3, 1, 2)
            v_pred_ = v_pred_.reshape(
                num_imgs,
                self.config.image_latent_dim,
                T,
                h_ * self.config.patch_size,
                w_ * self.config.patch_size,
            )

            return logits, v_pred_

    def _prepare_input(
        self,
        input_embeds,
        image_embeds,
        image_labels,
        image_masks,
        new_image_labels,
        modality_positions,
        height_embeds_proj,
        width_embeds_proj,
        time_embeds_proj,
    ):
        # Vision token format: <BOI><height><width><time><img><img>...<img><EOI>
        for i, modality_batch in enumerate(modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if offset < 0 and length < 0:
                    continue
                if self.config.add_time_embeds and self.config.add_aspect_ratio_embeds:
                    input_embeds[i, offset] = height_embeds_proj[
                        i * modality_positions.size(1) + j
                    ]
                    input_embeds[i, offset + 1] = width_embeds_proj[
                        i * modality_positions.size(1) + j
                    ]
                    input_embeds[i, offset + 2] = time_embeds_proj[
                        i * modality_positions.size(1) + j
                    ]
                    input_embeds[i, offset + 3 : offset + length] = image_embeds[
                        i * modality_positions.size(1) + j, : max(length - 3, 0)
                    ]
                    if image_labels is not None:
                        image_masks[i, offset] = 0
                        image_masks[i, offset + 1] = 0
                        image_masks[i, offset + 2] = 0
                        new_image_labels[i, offset + 3 : offset + length] = (
                            image_labels[
                                i * modality_positions.size(1) + j, : max(length - 3, 0)
                            ]
                        )
                elif self.config.add_time_embeds:
                    input_embeds[i, offset] = time_embeds_proj[
                        i * modality_positions.size(1) + j
                    ]
                    input_embeds[i, offset + 1 : offset + 1 + length - 1] = (
                        image_embeds[
                            i * modality_positions.size(1) + j, : max(length - 1, 0)
                        ]
                    )
                    if image_labels is not None:
                        image_masks[i, offset] = 0
                        new_image_labels[i, offset + 1 : offset + 1 + length - 1] = (
                            image_labels[
                                i * modality_positions.size(1) + j, : max(length - 1, 0)
                            ]
                        )
                else:
                    input_embeds[i, offset : offset + length] = image_embeds[
                        i * modality_positions.size(1) + j, :length
                    ]
                    if image_labels is not None:
                        new_image_labels[i, offset : offset + length] = image_labels[
                            i * modality_positions.size(1) + j, :length
                        ]
        return input_embeds, new_image_labels, image_masks

    @torch.no_grad()
    def t2i_generate(
        self,
        image_latents=None,
        t=None,
        text_tokens=None,
        attention_mask=None,
        diffhead_attention_mask=None,
        modality_positions=None,
        first_frame_as_cond=False,
        only_denoise_last_image=False,
        max_seq_len=None,
        guidance_scale=0.0,
        label=None,
        image_masks=None,
        image_labels=None,
        **kwargs,
    ):
        if guidance_scale > 0.0:
            if t.shape[-1] != text_tokens.shape[0]:
                t_cond, t_uncond = torch.chunk(t, 2)
                t_cond[:-1] = 1.0
                t_uncond[:-1] = 1.0
                t = torch.cat([t_cond, t_uncond])
            _, v = self(
                text_tokens,
                image_latents=image_latents,
                t=t,
                attention_mask=attention_mask,
                diffhead_attention_mask=diffhead_attention_mask,
                modality_positions=modality_positions,
                first_frame_as_cond=first_frame_as_cond,
                only_denoise_last_image=only_denoise_last_image,
                guidance_scale=guidance_scale,
                output_hidden_states=True,
                max_seq_len=max_seq_len,
            )
            v_cond, v_uncond = torch.chunk(v, 2)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            return torch.cat([v, v], dim=0)

    @torch.no_grad()
    def mmu_generate(
        self,
        input_embeds=None,
        attention_mask=None,
        max_new_tokens=100,
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_token=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        device = input_embeds.device

        result = []
        idx_next_embeds = input_embeds
        for i in range(max_new_tokens):
            if i == 0:
                model_output = self.showo(
                    inputs_embeds=input_embeds, attention_mask=attention_mask
                )
                logits = model_output.logits
                past_key_values = model_output.past_key_values
            else:
                model_output = self.showo(
                    inputs_embeds=idx_next_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = model_output.logits
                past_key_values = model_output.past_key_values

            if isinstance(attention_mask, BlockMask):
                attention_mask = step_block_mask_from_old(
                    attention_mask, attention_mask.seq_lengths[1] + 1
                )
            else:
                attention_mask = attention_mask.squeeze([0, 1])
                attention_mask = torch.hstack(
                    [attention_mask[-1, :], torch.tensor([0]).to(device)]
                ).unsqueeze(0)
                attention_mask = attention_mask.expand(1, 1, -1, -1)

            if not do_sample:
                idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # Apply top-p (nucleus) sampling
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                        :, :-1
                    ].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    logits[sorted_indices[sorted_indices_to_remove]] = -float("Inf")

                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            result.append(idx_next[0][0])
            # append sampled index to the running sequence and continue
            idx_next_embeds = self.showo.model.embed_tokens(idx_next)
            input_embeds = torch.cat([input_embeds, idx_next_embeds], dim=1)

            if eos_token is not None and idx_next.cpu() == eos_token:
                break

        return result
