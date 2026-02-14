import gc
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FluxPipeline
from omegaconf import DictConfig
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from data import create_indexed_dataloader
from matrix import create_matrix
from models import create_backbone_model, create_modified_head
from prototypes import compute_activation_bbox, pixelwise_multiply, topk_active_channels

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epic_purity(features, U, target_channels):
    B, C, H, W = features.shape
    rotated = pixelwise_multiply(features.to(torch.float32), U.to(torch.float32))
    batch_indices = torch.arange(B, device=features.device)
    target_map = rotated[batch_indices, target_channels]

    flat_map = target_map.view(B, -1)
    max_act, max_idx = torch.max(flat_map, dim=-1)

    max_y = max_idx // W
    max_x = max_idx % W

    all_channel_values = rotated[batch_indices, :, max_y, max_x]

    l2_norm = torch.linalg.norm(all_channel_values, dim=-1).clamp_min(1e-8)

    purity = max_act / l2_norm

    return purity, max_act, target_map


def decode_latents(pipeline, latents, height, width):
    lv = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    lv = (lv / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    return pipeline.vae.decode(
        lv.to(device=latents.device, dtype=pipeline.vae.dtype)
    ).sample


class LearnedPromptBank(nn.Module):
    def __init__(self, num_channels, pipe, device):
        super().__init__()
        self.num_channels = num_channels
        self.device = device

        with torch.no_grad():
            prompt = ""
            pe, ppe, _ = pipe.encode_prompt(
                prompt=[prompt], prompt_2=None, device=device
            )

        pe_init = pe.repeat(num_channels, 1, 1)
        pe_init = pe_init + torch.randn_like(pe_init) * 0.05

        ppe_init = ppe.repeat(num_channels, 1)
        ppe_init = ppe_init + torch.randn_like(ppe_init) * 0.05

        self.register_buffer("pe_anchor", pe_init.clone())
        self.register_buffer("ppe_anchor", ppe_init.clone())
        self.prompt_embeds = nn.Parameter(pe_init.clone())
        self.pooled_embeds = nn.Parameter(ppe_init.clone())

    def forward(self, channel_indices):
        prompt_embeds_ch = self.prompt_embeds[channel_indices]
        pooled_embeds_ch = self.pooled_embeds[channel_indices]
        return (
            prompt_embeds_ch + torch.randn_like(prompt_embeds_ch) * 0.05,
            pooled_embeds_ch + torch.randn_like(pooled_embeds_ch) * 0.05,
            self.pe_anchor[channel_indices],
            self.ppe_anchor[channel_indices],
        )


def differentiable_flux_generate(
    pipe,
    prompt_embeds,
    pooled_prompt_embeds,
    num_steps=4,
    device="cuda",
    height=256,
    width=256,
    seed=None,
):
    batch_size = prompt_embeds.shape[0]
    num_channels_latents = pipe.transformer.config.in_channels // 4

    generator = torch.Generator(device=device).manual_seed(seed) if seed else None

    latents, img_ids = pipe.prepare_latents(
        batch_size=batch_size,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    latents = latents.detach()

    text_seq_len = prompt_embeds.shape[1]
    txt_ids = torch.zeros(text_seq_len, 3, device=device, dtype=torch.bfloat16)

    image_seq_len = (height // 16) * (width // 16)
    use_dynamic_shifting = getattr(pipe.scheduler.config, "use_dynamic_shifting", False)
    mu = None
    if use_dynamic_shifting:
        base_seq_len = pipe.scheduler.config.get("base_image_seq_len", 256)
        max_seq_len = pipe.scheduler.config.get("max_image_seq_len", 4096)
        base_shift = pipe.scheduler.config.get("base_shift", 0.5)
        max_shift = pipe.scheduler.config.get("max_shift", 1.15)
        m = (image_seq_len - base_seq_len) / (max_seq_len - base_seq_len)
        mu = base_shift + m * (max_shift - base_shift)

    if mu is not None:
        pipe.scheduler.set_timesteps(num_steps, device=device, mu=mu)
    else:
        pipe.scheduler.set_timesteps(num_steps, device=device)
    guidance_vec = torch.full((batch_size,), 3.5, device=device, dtype=torch.bfloat16)

    for t in pipe.scheduler.timesteps:
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=t.expand(batch_size).to(device) / 1000.0,
            guidance=guidance_vec,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    final_img = decode_latents(pipe, latents, height, width)
    return (final_img / 2 + 0.5).clamp(0, 1)


def run_epic_generative(config: DictConfig):
    set_seeds(config.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_bundle = create_backbone_model(
        config.model.name,
        device,
        config.model.custom_weights_path,
        config.model.num_classes,
    )
    backbone = model_bundle.backbone.to(torch.float32).eval()
    base_model = model_bundle.base_model.to(torch.float32).eval()
    backbone.requires_grad_(False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    m_3d, s_3d = mean.squeeze().view(3, 1, 1).cpu(), std.squeeze().view(3, 1, 1).cpu()

    print("Loading Flux Pipeline...")
    pipe = FluxPipeline.from_pretrained(
        config.generative_model.model_id, torch_dtype=torch.bfloat16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)

    pipe.transformer.enable_gradient_checkpointing()
    pipe.vae.enable_gradient_checkpointing()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    U = create_matrix(config.matrix.type, model_bundle.num_channels, device).to(
        torch.float32
    )

    prompt_bank = LearnedPromptBank(model_bundle.num_channels, pipe, device).to(
        torch.bfloat16
    )

    opt_prompts = torch.optim.AdamW(prompt_bank.parameters(), lr=config.training.lr_reg)
    opt_U = torch.optim.AdamW(U.parameters(), lr=config.training.lr_purity)

    purity_history = []
    os.makedirs(config.output_path, exist_ok=True)
    grid_dir = os.path.join(config.output_path, "periodic_grids")
    os.makedirs(grid_dir, exist_ok=True)

    val_loader = create_indexed_dataloader(
        config.dataset.datapath_val,
        model_bundle.transform,
        1,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        shuffle=True,
    )
    val_iter = iter(val_loader)

    pbar = tqdm(range(config.training.steps), desc="Learning Concepts")

    for step in pbar:
        B = config.training.batch_size
        target_channels = torch.randint(
            0, model_bundle.num_channels, (B,), device=device
        )

        U.requires_grad_(False)

        pe, ppe, pea, ppea = prompt_bank(target_channels)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_imgs = differentiable_flux_generate(
                pipe, pe, ppe, num_steps=config.training.gen_steps, device=device
            )
            imgs_in = F.interpolate(generated_imgs, (224, 224), mode="bilinear")
            feats = backbone((imgs_in.float() - mean) / std)
            purity_scores, max_act, t_map = epic_purity(feats, U(), target_channels)
            loss_purity = -config.training.lambda_purity * purity_scores.mean()
            loss_reg = config.training.lambda_reg * (
                F.mse_loss(pe, pea) + F.mse_loss(ppe, ppea)
            )
            loss_act = -0.5 * torch.log(max_act + 1e-6).mean()
            loss_sparse = (
                -0.1 * (max_act / t_map.view(1, -1).mean(dim=1).clamp_min(1e-6)).mean()
            )
            total_loss = loss_purity + loss_reg + loss_act + loss_sparse

        opt_prompts.zero_grad(set_to_none=True)

        total_loss.backward()

        opt_prompts.step()
        if step % 2 == 0:
            U.requires_grad_(True)
            opt_U.zero_grad(set_to_none=True)
            with torch.no_grad():
                feats_static = feats.detach()
            p_u, _, _ = epic_purity(feats_static, U(), target_channels)
            (-config.training.lambda_purity * p_u.mean()).backward()
            opt_U.step()

        purity_history.append(purity_scores.mean().item())

        if step % 50 == 0 or (step + 1) == config.training.steps:
            plt.figure(figsize=(10, 6))
            plt.plot(purity_history, color="tab:blue")
            plt.title(f"EPIC Purity Progress (Iter {step + 1})")
            plt.savefig(os.path.join(config.output_path, "purity_history.png"))
            plt.close()

            try:
                img_v, _ = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                img_v, _ = next(val_iter)

            img_v_cuda = img_v.to(device)
            mod_head = create_modified_head(base_model, config.model.name, U=U)

            with torch.no_grad():
                v_feats_raw = backbone((img_v_cuda - mean) / std)
                v_feats_rot = pixelwise_multiply(
                    v_feats_raw.to(torch.float32),
                    U().to(torch.float32),
                )

            top_ch = topk_active_channels(
                backbone, mod_head, img_v_cuda[0], k=4, device=device
            )
            fig, axes = plt.subplots(4, 6, figsize=(24, 16))

            for row, ch in enumerate(top_ch):
                orig_img_tensor = (img_v_cuda[0].cpu() * s_3d + m_3d).clip(0, 1)
                orig_pil = to_pil_image(orig_img_tensor)
                x1, y1, x2, y2 = compute_activation_bbox(
                    v_feats_rot[0, ch].unsqueeze(0),
                    (orig_img_tensor.shape[1], orig_img_tensor.shape[2]),
                )
                ImageDraw.Draw(orig_pil).rectangle(
                    [x1.item(), y1.item(), x2.item() - 1, y2.item() - 1],
                    outline=(255, 255, 0),
                    width=3,
                )

                axes[row, 0].imshow(np.array(orig_pil))
                axes[row, 0].set_ylabel(f"Channel {ch}", fontsize=14, fontweight="bold")
                axes[row, 0].set_xticks([])
                axes[row, 0].set_yticks([])

                ch_tensor = torch.tensor([ch] * 5, device=device)

                with torch.no_grad():
                    pe_vis, ppe_vis, _, _ = prompt_bank(ch_tensor)

                    ex_imgs_t = differentiable_flux_generate(
                        pipe,
                        pe_vis,
                        ppe_vis,
                        num_steps=config.training.gen_steps,
                        device=device,
                    )
                    ex_imgs = ex_imgs_t.float().cpu().permute(0, 2, 3, 1).numpy()

                for col in range(5):
                    proto_t = (
                        torch.from_numpy(ex_imgs[col])
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(device, torch.float32)
                    )
                    proto_in = F.interpolate(proto_t, (224, 224), mode="bilinear")

                    with torch.no_grad():
                        proto_feats = backbone((proto_in - mean) / std)
                        proto_rot = pixelwise_multiply(
                            proto_feats.to(torch.float32),
                            U().to(torch.float32),
                        )

                    px1, py1, px2, py2 = compute_activation_bbox(
                        proto_rot[0, ch].unsqueeze(0), (256, 256)
                    )

                    proto_pil = to_pil_image(
                        torch.from_numpy(ex_imgs[col]).permute(2, 0, 1)
                    )
                    ImageDraw.Draw(proto_pil).rectangle(
                        [px1.item(), py1.item(), px2.item() - 1, py2.item() - 1],
                        outline=(255, 255, 0),
                        width=3,
                    )

                    axes[row, col + 1].imshow(np.array(proto_pil))
                    axes[row, col + 1].axis("off")

            plt.tight_layout()
            plt.savefig(
                os.path.join(grid_dir, f"step_{step + 1:04d}.jpg"), bbox_inches="tight"
            )
            plt.close()
            gc.collect()
            torch.cuda.empty_cache()

        if step % 5 == 0:
            pbar.set_postfix(
                {
                    "purity": f"{purity_scores.mean().item():.4f}",
                    "reg": f"{loss_reg.item():.4f}",
                }
            )

    U.save_state(config.output_path, f"{config.matrix.type}.pt")
    torch.save(
        prompt_bank.state_dict(), os.path.join(config.output_path, "learned_prompts.pt")
    )
