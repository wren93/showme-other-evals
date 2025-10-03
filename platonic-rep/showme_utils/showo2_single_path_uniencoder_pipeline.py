# pyre-unsafe

from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from showme_utils.misc import prepare_gen_input
from showme_utils.omni_attention import omni_attn_mask_naive


def denorm(images):
    """
    Denormalize images from [-1, 1] to [0, 255] and convert to numpy arrays.

    Args:
        images: Tensor of shape (B, C, H, W) with values in [-1, 1]

    Returns:
        Numpy array of shape (B, H, W, C) with values in [0, 255]
    """
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return images


def get_hyper_params(
    text_tokenizer,
    showo_token_ids,
    use_chat_template=False,
    add_aspect_ratio_embeds=False,
    height=512,
    width=512,
    generation_mode="t2i",
    latent_frames=1,
):
    """
    Extract hyperparameters from config.

    Args:
        config: Configuration object
        text_tokenizer: Text tokenizer
        showo_token_ids: Showo token IDs dictionary

    Returns:
        Tuple of hyperparameters
    """
    # Extract basic parameters
    if width == "auto":
        width = 512
    if height == "auto":
        height = 512
    latent_width = width // 16
    latent_height = height // 16
    num_image_tokens = (
        latent_width * latent_height + int(add_aspect_ratio_embeds) * 2 + 1
    )
    num_video_tokens = (
        latent_width * latent_height * latent_frames
        + int(add_aspect_ratio_embeds) * 2
        + 1
    )
    if generation_mode == "t2i":
        max_seq_len = 2048
        max_text_len = (
            2048 - num_image_tokens - 33
            if use_chat_template
            else 2048 - num_image_tokens - 4
        )
    else:
        max_seq_len = num_video_tokens - 1 + 1024
        max_text_len = (
            max_seq_len - num_video_tokens - 33
            if use_chat_template
            else max_seq_len - num_video_tokens - 4
        )
    image_latent_dim = 48
    patch_size = 1

    # Token IDs
    pad_id = text_tokenizer.pad_token_id
    bos_id = showo_token_ids["bos_id"]
    eos_id = showo_token_ids["eos_id"]
    boi_id = showo_token_ids["boi_id"]
    eoi_id = showo_token_ids["eoi_id"]
    bov_id = showo_token_ids["bov_id"]
    eov_id = showo_token_ids["eov_id"]
    img_pad_id = showo_token_ids["img_pad_id"]
    vid_pad_id = showo_token_ids["vid_pad_id"]

    # Guidance scale
    guidance_scale = 7.5
    return (
        num_image_tokens,
        num_video_tokens,
        max_seq_len,
        max_text_len,
        image_latent_dim,
        patch_size,
        latent_width,
        latent_height,
        pad_id,
        bos_id,
        eos_id,
        boi_id,
        eoi_id,
        bov_id,
        eov_id,
        img_pad_id,
        vid_pad_id,
        guidance_scale,
    )


def prepare_gen_input_chat(
    prompts,
    text_tokenizer,
    num_image_tokens,
    bos_id,
    eos_id,
    boi_id,
    eoi_id,
    pad_id,
    img_pad_id,
    max_text_len,
    max_seq_len,
    device,
):
    batch_text_tokens = []
    batch_modality_positions = []
    batch_text_tokens_null = []
    batch_modality_positions_null = []
    for prompt in prompts:
        text_tokens = text_tokenizer(prompt, add_special_tokens=False)["input_ids"][
            :(max_text_len)
        ]
        prompt = text_tokenizer.decode(text_tokens)

        conversation = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "<image>"},
        ]
        conv_prompt = text_tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        text_tokens = text_tokenizer(conv_prompt, add_special_tokens=False).input_ids
        img_id = text_tokenizer("<image>", add_special_tokens=False).input_ids[0]
        img_idx = text_tokens.index(img_id)

        modality_positions = torch.tensor([img_idx + 1, num_image_tokens]).unsqueeze(0)
        text_tokens = (
            text_tokens[:img_idx]
            + [boi_id]
            + [img_pad_id] * num_image_tokens
            + [eoi_id]
            + text_tokens[img_idx + 1 :]
        )
        text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
        batch_text_tokens.append(torch.tensor(text_tokens))
        batch_modality_positions.append(modality_positions)
        ##### original
        # text_tokens_null = []
        ####
        text_tokens_null = text_tokenizer(
            "ugly, distorted, deformed, disfigured, low quality, worst quality, blurry, noisy, pixelated, overexposed, underexposed, bad anatomy, bad proportions, extra limbs, missing limbs, fused fingers, extra fingers, poorly drawn hands, poorly drawn face, asymmetrical eyes, messed up face, disfigured mouth, unnatural lighting, strange reflections, artifact, jpeg artifacts, watermark, text, subtitle, logo, frame border, over-saturated, color bleeding, unrealistic colors, low-res, low resolution, bad composition, messy background, cluttered, cropped head, cut-off body, unnatural pose, broken limbs, wrong perspective, out of frame, duplicated parts",
            add_special_tokens=False,
        )["input_ids"][:(max_text_len)]
        prompt_null = text_tokenizer.decode(text_tokens_null)
        conversation_null = conversation.copy()
        conversation_null[1]["content"] = prompt_null
        conv_prompt_null = text_tokenizer.apply_chat_template(
            conversation_null, tokenize=False, add_generation_prompt=False
        )
        text_tokens_null = text_tokenizer(
            conv_prompt_null, add_special_tokens=False
        ).input_ids

        img_idx = text_tokens_null.index(img_id)
        modality_positions_null = torch.tensor(
            [img_idx + 1, num_image_tokens]
        ).unsqueeze(0)
        text_tokens_null = (
            text_tokens_null[:img_idx]
            + [boi_id]
            + [img_pad_id] * num_image_tokens
            + [eoi_id]
            + text_tokens_null[img_idx + 1 :]
        )
        text_tokens_null = text_tokens_null + [pad_id] * (
            max_seq_len - len(text_tokens_null)
        )

        batch_text_tokens_null.append(torch.tensor(text_tokens_null))
        batch_modality_positions_null.append(modality_positions_null)

    batch_text_tokens = torch.stack(batch_text_tokens, dim=0).to(device)
    batch_modality_positions = torch.stack(batch_modality_positions, dim=0).to(device)

    batch_text_tokens_null = torch.stack(batch_text_tokens_null, dim=0).to(device)
    batch_modality_positions_null = torch.stack(
        batch_modality_positions_null, dim=0
    ).to(device)

    return (
        batch_text_tokens,
        batch_text_tokens_null,
        batch_modality_positions,
        batch_modality_positions_null,
    )


# Path to LLM name mapping
path_to_llm_name = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen2_5",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2_5",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2_5",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen2_5",
}


class Showo2SinglePathUniEncoderPipeline:
    """
    Pipeline for text-to-image generation using Showo2 model.
    Similar structure to SimpleVtonPipeline but adapted for Showo2's single-path architecture.
    """

    def __init__(
        self,
        model,
        vae_model,
        text_tokenizer=None,
        showo_token_ids=None,
        config=None,
        weight_dtype=torch.float32,
        device="cuda",
        use_tf32=True,
        use_chat_template=True,
        add_aspect_ratio_embeds=False,
        height=512,
        width=512,
        latent_frames=1,
        generation_mode="t2i",
    ):
        self.model = model
        self.latent_frames = latent_frames
        self.generation_mode = generation_mode
        self.vae_model = vae_model
        self.text_tokenizer = text_tokenizer
        self.showo_token_ids = showo_token_ids
        self.config = config
        self.device = device
        self.weight_dtype = weight_dtype
        self.use_chat_template = use_chat_template
        self.add_aspect_ratio_embeds = add_aspect_ratio_embeds
        self.num_visual_tokens = 1009
        self.height = height
        self.width = width
        self.conversation = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
        ]

        # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

        # Initialize hyperparameters if config is provided
        if text_tokenizer and showo_token_ids:
            self._init_hyperparams(self.height, self.width)

    def _init_hyperparams(self, height, width):
        """Initialize hyperparameters from config"""
        (
            self.num_image_tokens,
            self.num_video_tokens,
            self.max_seq_len,
            self.max_text_len,
            self.image_latent_dim,
            self.patch_size,
            self.latent_width,
            self.latent_height,
            self.pad_id,
            self.bos_id,
            self.eos_id,
            self.boi_id,
            self.eoi_id,
            self.bov_id,
            self.eov_id,
            self.img_pad_id,
            self.vid_pad_id,
            self.guidance_scale,
        ) = get_hyper_params(
            self.text_tokenizer,
            self.showo_token_ids,
            self.use_chat_template,
            self.add_aspect_ratio_embeds,
            height,
            width,
            self.generation_mode,
            self.latent_frames,
        )
        if self.generation_mode == "t2i":
            self.num_visual_tokens = self.num_image_tokens
        else:
            self.num_visual_tokens = self.num_video_tokens

    @torch.no_grad()
    def t2i(
        self,
        prompts: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        transport=None,
        sampler=None,
        sampling_method: str = "euler",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
        time_shifting_factor: float = 3.0,
        noise_level: float = 1.0,
        input_latents: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate images from text prompts.

        Args:
            prompts: Text prompt(s) for image generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            generator: Random generator for reproducible generation
            transport: Transport object for sampling
            sampler: Sampler object
            sampling_method: Sampling method (e.g., "dopri5", "euler")
            atol: Absolute tolerance for ODE solver
            rtol: Relative tolerance for ODE solver
            reverse: Whether to reverse the sampling process
            time_shifting_factor: Time shifting factor for sampling
            noise_level: Noise level to start from (0.0=clean, 1.0=pure noise, 0.5=halfway)
            input_latents: Optional input latents to add noise to. If None, uses random noise.

        Returns:
            List of generated PIL images
        """
        # Handle single prompt
        if isinstance(prompts, str):
            prompts = [prompts]

        batch_size = len(prompts)

        # Prepare text tokens and modality positions
        if self.use_chat_template:
            (
                batch_text_tokens,
                batch_text_tokens_null,
                batch_modality_positions,
                batch_modality_positions_null,
            ) = prepare_gen_input_chat(
                prompts,
                self.text_tokenizer,
                self.num_visual_tokens,
                self.bos_id,
                self.eos_id,
                self.boi_id,
                self.eoi_id,
                self.pad_id,
                self.img_pad_id,
                self.max_text_len,
                self.max_seq_len,
                self.device,
            )
        else:
            (
                batch_text_tokens,
                batch_text_tokens_null,
                batch_modality_positions,
                batch_modality_positions_null,
            ) = prepare_gen_input(
                prompts,
                self.text_tokenizer,
                self.num_visual_tokens,
                self.bos_id,
                self.eos_id,
                self.boi_id,
                self.eoi_id,
                self.pad_id,
                self.img_pad_id,
                self.max_text_len,
                self.device,
            )

        if sampler is not None and transport is not None:
            sample_fn, t_start = sampler.sample_ode(
                sampling_method="euler",
                num_steps=num_inference_steps,
                atol=atol,
                rtol=rtol,
                reverse=reverse,
                time_shifting_factor=time_shifting_factor,
                noise_level=noise_level,
            )
        # Initialize latents with controlled noise level
        if input_latents is not None:
            # Use provided latents as starting point
            x1 = input_latents.to(self.weight_dtype).to(self.device)
            if x1.shape[0] != batch_size:
                x1 = x1.repeat(batch_size, 1, 1, 1)

        z = (
            torch.randn(
                (
                    batch_size,
                    self.image_latent_dim,
                    self.latent_frames,
                    self.latent_height * self.patch_size,
                    self.latent_width * self.patch_size,
                )
            )
            .to(self.weight_dtype)
            .to(self.device)
        )

        # Prepare inputs for classifier-free guidance
        if guidance_scale > 0:
            z = torch.cat([z, z], dim=0)
            text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
            modality_positions = torch.cat(
                [batch_modality_positions, batch_modality_positions_null], dim=0
            )

            # Create attention mask
            block_mask = omni_attn_mask_naive(
                text_tokens.size(0), self.max_seq_len, modality_positions, self.device
            ).to(self.weight_dtype)
        else:
            text_tokens = batch_text_tokens
            modality_positions = batch_modality_positions

            # Create attention mask
            block_mask = omni_attn_mask_naive(
                text_tokens.size(0), self.max_seq_len, modality_positions, self.device
            ).to(self.weight_dtype)

        model_kwargs = {
            "text_tokens": text_tokens,
            "attention_mask": block_mask,
            "modality_positions": modality_positions,
            "output_hidden_states": True,
            "max_seq_len": self.max_seq_len,
            "guidance_scale": guidance_scale,
        }

        # Sample using transport

        samples = sample_fn(  # pyre-ignore
            z, self.model.showo_model.t2i_generate, **model_kwargs
        )[-1]

        # Handle classifier-free guidance
        if guidance_scale > 0:
            samples = torch.chunk(samples, 2)[0]
        # Decode latents to images
        images = self._decode_latents(samples)
        return images

    def _decode_latents(self, latents: torch.Tensor) -> List[Image.Image]:
        """
        Decode latents to PIL images using VAE model.

        Args:
            latents: Latent representations to decode

        Returns:
            List of PIL images
        """
        if hasattr(self.vae_model, "batch_decode"):
            # For WanVAE or similar models
            if len(latents.shape) == 4:
                latents = latents.unsqueeze(2)  # Add temporal dimension
            images = self.vae_model.batch_decode(latents)
            if len(images.shape) == 5:
                images = images.squeeze(2)  # Remove temporal dimension
        else:
            device, dtype = latents.device, latents.dtype
            scale = self.model.showo_model.vision_model.get_vae_scale(device, dtype)
            images = self.vae_model.decode(latents, scale=scale)

            if images.shape[2] == 1:
                images = images.squeeze(2)

        # Convert to PIL images
        # import ipdb

        # ipdb.set_trace()
        if self.generation_mode == "t2i":
            images = denorm(images)
            pil_images = [Image.fromarray(image) for image in images]
            return pil_images
        else:
            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).to(
                torch.float32
            )
            images *= 255.0
            images = (
                images.permute(0, 2, 3, 4, 1).cpu().numpy().astype(np.uint8)
            )  # [B, T, H, W, C]

            frames = [images]

            return frames  # pyre-ignore

    def mmu(
        self,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_new_tokens: int = 512,
        prompt: str = "",
        pixel_values: Optional[torch.Tensor] = None,
        height: int = 512,
        width: int = 512,
    ) -> List[str]:
        self._init_hyperparams(height, width)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
            prompt = "<image>\n" + prompt
        conversation = self.conversation.copy()
        conversation.append({"role": "user", "content": prompt})
        conv_prompt = self.text_tokenizer.apply_chat_template(  # pyre-ignore
            conversation, tokenize=False, add_generation_prompt=True
        )
        text_tokens = self.text_tokenizer(  # pyre-ignore
            conv_prompt, add_special_tokens=False
        ).input_ids
        img_id = self.text_tokenizer("<image>", add_special_tokens=False).input_ids[0]  # pyre-ignore
        img_idx = text_tokens.index(img_id)
        text_tokens = (
            text_tokens[:img_idx]
            + [self.boi_id]
            + [self.img_pad_id] * self.num_image_tokens
            + [self.eoi_id]
            + text_tokens[img_idx + 1 :]
        )
        text_tokens = torch.tensor(text_tokens).unsqueeze(0).to(self.device)
        modality_positions = (
            torch.tensor([[img_idx + 1, self.num_image_tokens]])
            .unsqueeze(0)
            .to(self.device)
        )
        text_masks = torch.where(
            # pyre-ignore[6]
            (text_tokens != self.img_pad_id) & (text_tokens != self.pad_id),
            torch.ones_like(text_tokens),
            torch.zeros_like(text_tokens),
        )
        image_masks = torch.where(
            text_tokens == self.img_pad_id,  # pyre-ignore[6]
            torch.ones_like(text_tokens),
            torch.zeros_like(text_tokens),
        ).to(self.device)
        data_type = ["mmu"]

        image_latents, t, image_labels, image_masks = (
            self.model.prepare_latents_and_labels(pixel_values, data_type, image_masks)
        )
        block_mask = self.model.create_attention_mask(
            text_tokens.size(0),
            text_tokens.size(1),
            modality_positions,
            self.device,
            self.weight_dtype,
        )

        model_output = self.model.showo_model(
            text_tokens=text_tokens,
            image_latents=image_latents,
            t=t.to(self.weight_dtype),
            attention_mask=block_mask,
            text_masks=text_masks,
            image_masks=image_masks,
            text_labels=None,
            image_labels=image_labels,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=text_tokens.size(1),
            device=text_tokens.device,
            return_input_embeds=True,
        )

        if self.model.mrope_type == "none" or self.model.mrope_type == "dit_rope":
            input_embeds = model_output
            output_tokens = self.model.showo_model.mmu_generate(
                input_embeds=input_embeds,
                attention_mask=block_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token=self.text_tokenizer.eos_token_id,  # pyre-ignore
            )
        else:
            input_embeds, position_ids = model_output
            output_tokens = self.model.showo_model.mmu_generate(
                input_embeds=input_embeds,
                position_ids=position_ids,
                attention_mask=block_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token=self.text_tokenizer.eos_token_id,  # pyre-ignore
            )

        text = self.text_tokenizer.decode(output_tokens, skip_special_tokens=True)  # pyre-ignore

        return text
