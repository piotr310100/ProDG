import gc
import os
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline
from omegaconf import DictConfig
from tqdm import tqdm

from data import create_indexed_dataloader
from matrix import create_matrix
from models import create_backbone_model, create_modified_head
from prototypes import topk_active_channels

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def total_variation(x):
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.mean(
        torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    )


def epic_purity(features, U, target_channels):
    B, C, H, W = features.shape
    rotated = torch.einsum(
        "bchw,cd->bdhw", features.to(torch.float32), U.to(torch.float32)
    )
    flat = rotated.view(B, C, -1)
    max_vals, _ = flat.max(dim=2)
    l2 = torch.norm(flat, dim=2).clamp_min(1e-6)
    return (
        max_vals[torch.arange(B), target_channels]
        / l2[torch.arange(B), target_channels]
    )


def pack_latents(latents):
    batch_size, num_channels, height, width = latents.shape
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels * 4
    )
    return latents


def unpack_latents(latents, height, width):
    batch_size, num_patches, channels = latents.shape
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, height, width)
    return latents


def prepare_flux_img_ids(height_latent, width_latent, device, dtype):
    h_ids = height_latent // 2
    w_ids = width_latent // 2

    h_grid = torch.arange(h_ids, device=device)
    w_grid = torch.arange(w_ids, device=device)

    grid_h, grid_w = torch.meshgrid(h_grid, w_grid, indexing="ij")

    grid_h = grid_h.reshape(-1)
    grid_w = grid_w.reshape(-1)

    img_ids = torch.stack([grid_h, grid_w, torch.zeros_like(grid_h)], dim=-1)
    return img_ids.to(dtype=dtype)


def generate_guided_prototypes(
    pipe,
    backbone,
    U,
    channel_idx,
    device,
    mean,
    std,
    n=1,
    num_steps=20,
    guidance_scale=2000.0,
    seed=None,
):
    if seed is not None:
        generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    height, width = 512, 512
    target_ch = torch.full((n,), channel_idx, device=device)
    prompt = "a high quality detailed photo"

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=[prompt] * n, prompt_2=None, device=device
        )
        if text_ids.ndim == 3:
            text_ids = text_ids[0]

    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents = torch.randn(
        (n, num_channels_latents, height // 8, width // 8),
        device=device,
        dtype=pipe.dtype,
        generator=generator,
    )

    img_ids = prepare_flux_img_ids(height // 8, width // 8, device, pipe.dtype)

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(num_steps, device=device)

    for i, t in enumerate(scheduler.timesteps):
        current_guidance = guidance_scale * (1 - i / num_steps)

        if current_guidance > 1.0:
            latents_grad = latents.detach().clone().requires_grad_(True)

            with torch.no_grad():
                packed_input = pack_latents(latents_grad)
                t_expand = t.expand(n).to(device)

                noise_pred_packed = pipe.transformer(
                    hidden_states=packed_input,
                    timestep=t_expand / 1000.0,
                    guidance=None,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]

                noise_pred = unpack_latents(noise_pred_packed, height // 8, width // 8)

            t_frac = (t / 1000.0).to(latents_grad.dtype)
            x0_latents = latents_grad - t_frac * noise_pred

            with torch.amp.autocast("cuda", enabled=False):
                img = pipe.vae.decode(
                    x0_latents.float() / pipe.vae.config.scaling_factor
                ).sample

            img_01 = (img.float() / 2 + 0.5).clamp(0, 1)

            x_in = F.interpolate(
                img_01, (224, 224), mode="bilinear", align_corners=False
            )
            feats = backbone((x_in - mean) / std)
            purity = epic_purity(feats, U, target_ch)

            loss = -purity.mean() + 0.05 * total_variation(img_01)
            grad = torch.autograd.grad(loss, latents_grad)[0]

            with torch.no_grad():
                grad_norm = grad.norm(p=2, dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
                shift = -current_guidance * (grad / grad_norm)
                latents = latents + shift.to(latents.dtype)

            del img, img_01, x0_latents, grad, latents_grad
            torch.cuda.empty_cache()

        with torch.no_grad():
            packed_input = pack_latents(latents)
            t_expand = t.expand(n).to(device)

            noise_pred_packed = pipe.transformer(
                hidden_states=packed_input,
                timestep=t_expand / 1000.0,
                guidance=None,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]

            noise_pred = unpack_latents(noise_pred_packed, height // 8, width // 8)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            final_img = pipe.vae.decode(
                latents.float() / pipe.vae.config.scaling_factor
            ).sample

    return (final_img.float() / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()


def run_epic_generative(config: DictConfig):
    set_seeds(config.seed)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    model_bundle = create_backbone_model(
        config.model.name,
        device,
        config.model.custom_weights_path,
        config.model.num_classes,
    )
    backbone = torch.compile(model_bundle.backbone.to(torch.float32).eval())
    base_model = model_bundle.base_model.to(torch.float32).eval()
    num_channels = model_bundle.num_channels

    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(
        1, 3, 1, 1
    )
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(
        1, 3, 1, 1
    )

    pipe = FluxPipeline.from_pretrained(
        config.generative_model.model_id, torch_dtype=torch.bfloat16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.to(torch.float32)

    U = create_matrix(config.matrix.type, num_channels, device).to(torch.float32)
    optimizer = torch.optim.Adam(U.parameters(), lr=config.training.lr)

    pbar = tqdm(range(config.training.steps), desc="EPIC Training")
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

    for step in pbar:
        B = config.training.batch_size
        target_channels = torch.randint(0, num_channels, (B,), device=device)

        feats_list = []
        for i in range(B):
            img_np = generate_guided_prototypes(
                pipe,
                backbone,
                U().detach(),
                target_channels[i],
                device,
                mean,
                std,
                n=1,
                num_steps=config.training.gen_steps,
                guidance_scale=config.training.guidance_scale,
            )
            img_tensor = (
                torch.from_numpy(img_np).permute(0, 3, 1, 2).to(device, torch.float32)
            )
            x_in = F.interpolate(
                img_tensor, (224, 224), mode="bilinear", align_corners=False
            )
            with torch.no_grad():
                f = backbone((x_in - mean) / std)
            feats_list.append(f)

        all_feats = torch.cat(feats_list, dim=0)

        optimizer.zero_grad()
        purity_train = epic_purity(all_feats, U(), target_channels)
        loss_u = -purity_train.mean()
        loss_u.backward()
        optimizer.step()

        purity_history.append(purity_train.mean().item())

        if step % 50 == 0 or (step + 1) == config.training.steps:
            plt.figure(figsize=(10, 6))
            plt.plot(purity_history, color="tab:blue")
            plt.title(f"EPIC Purity Progress (Iter {step + 1})")
            plt.savefig(os.path.join(config.output_path, "purity_history.png"))
            plt.close()

            img_v, _ = next(iter(val_loader))
            head_tmp = create_modified_head(base_model, config.model.name, U)
            top_ch = topk_active_channels(
                backbone, head_tmp, img_v[0].to(device), k=4, device=device
            )

            fig, axes = plt.subplots(4, 6, figsize=(24, 16))
            orig_np = (
                img_v[0].permute(1, 2, 0).numpy() * [0.229, 0.224, 0.225]
                + [0.485, 0.456, 0.406]
            ).clip(0, 1)

            for row, ch in enumerate(top_ch):
                axes[row, 0].imshow(orig_np)
                axes[row, 0].set_ylabel(f"Ch {ch}", fontsize=14, fontweight="bold")
                ex_imgs = generate_guided_prototypes(
                    pipe,
                    backbone,
                    U().detach(),
                    ch,
                    device,
                    mean,
                    std,
                    n=5,
                    num_steps=20,
                    guidance_scale=config.training.guidance_scale,
                )
                for col in range(5):
                    axes[row, col + 1].imshow(ex_imgs[col])
                    axes[row, col + 1].axis("off")

            plt.savefig(
                os.path.join(grid_dir, f"step_{step + 1:04d}.jpg"), bbox_inches="tight"
            )
            plt.close()
            del all_feats, feats_list
            gc.collect()
            torch.cuda.empty_cache()

        if step % 5 == 0:
            pbar.set_postfix({"purity": f"{purity_train.mean().item():.4f}"})

    U.save_state(config.output_path, f"{config.matrix.type}.pt")