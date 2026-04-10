import gc
import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
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


@torch.no_grad()
def discover_initial_prompts(config, pipe, backbone, device):
    with open(config.dataset.class_map_json, "r") as f:
        class_map = json.load(f)

    class_names = list(class_map.values())
    dummy_feat = backbone(torch.randn(1, 3, 224, 224).to(device))
    num_channels = dummy_feat.shape[1]

    best_purities = torch.full((num_channels,), -1.0, device=device)
    best_names = [""] * num_channels

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    backbone.eval()
    gen_batch_size = config.generative_model.get("discovery_batch_size", 4)

    for i in tqdm(range(0, len(class_names), gen_batch_size), desc="Discovery"):
        batch_names = class_names[i : i + gen_batch_size]

        with torch.no_grad():
            pe, ppe, _ = pipe.encode_prompt(
                prompt=batch_names, prompt_2=None, device=device
            )
            gen_imgs = differentiable_flux_generate(
                pipe,
                pe,
                ppe,
                num_steps=config.generative_model.gen_steps,
                guidance_scale=config.generative_model.guidance_scale,
                device=device,
                seed=config.seed,
            )

            imgs_in = F.interpolate(gen_imgs, (224, 224), mode="bilinear")
            feats = backbone((imgs_in.float() - mean) / std).to(torch.float32)
            B, C, H, W = feats.shape

            flat_feats = feats.view(B, C, -1)
            max_act, max_idx = torch.max(flat_feats, dim=-1)
            batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, C)
            all_channel_vectors = flat_feats[batch_indices, :, max_idx]
            l2_norms = torch.linalg.norm(all_channel_vectors, dim=-1).clamp_min(1e-8)
            batch_purities = max_act / l2_norms

            current_best_batch_val, current_best_batch_idx = torch.max(
                batch_purities, dim=0
            )

            mask = current_best_batch_val > best_purities
            best_purities[mask] = current_best_batch_val[mask]

            for ch_idx in torch.where(mask)[0]:
                img_in_batch_idx = current_best_batch_idx[ch_idx]
                best_names[ch_idx.item()] = batch_names[img_in_batch_idx]

        torch.cuda.empty_cache()

    print(
        best_names,
        best_purities.min().item(),
        best_purities.mean().item(),
        best_purities.max().item(),
    )

    return best_names


def get_heatmap(
    model: nn.Module,
    modified_head: nn.Module,
    images: torch.Tensor,
    batch_size: int = 1,
    device: torch.device | str = "cuda",
) -> np.ndarray:
    # TODO: Add weighing to the box selection?
    with torch.inference_mode():
        image = images.to(device)
        concepts = modified_head._preprocess_input(model(image))
        concepts_relu = torch.nn.functional.relu(concepts)

        pixel_norms = torch.linalg.norm(concepts, dim=1, keepdim=True)
        purity = concepts_relu / (pixel_norms + 1e-8)

        B, C, H, W = concepts_relu.shape
        ch_max = concepts_relu.view(B, C, -1).max(dim=-1).values.view(B, C, 1, 1)
        magnitude = concepts_relu / (ch_max + 1e-8)

        heatmap = purity * magnitude

        heatmap_rescaled = (
            torch.nn.functional.interpolate(
                heatmap, images.shape[-2:], mode="bilinear", align_corners=False
            )
            .cpu()
            .numpy()
        )

    return heatmap_rescaled


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epic_purity(features, U, target_channels):
    B, C, H, W = features.shape
    rotated = pixelwise_multiply(features, U)
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


class VariationalPromptBank(nn.Module):
    def __init__(self, num_channels, pipe, device, initial_prompts: list[str] = None, rank = 128):
        super().__init__()
        self.num_channels = num_channels
        self.device = device

        if initial_prompts is not None:
            print(f"Encoding {len(initial_prompts)} discovered prompts...")
            with torch.no_grad():
                unique_prompts = list(set(initial_prompts))
                p_to_idx = {p: i for i, p in enumerate(unique_prompts)}

                all_pe, all_ppe = [], []
                for i in range(0, len(unique_prompts), 8):
                    batch = unique_prompts[i : i + 8]
                    pe, ppe, _ = pipe.encode_prompt(
                        prompt=batch, prompt_2=None, device=device
                    )
                    all_pe.append(pe)
                    all_ppe.append(ppe)

                u_pe = torch.cat(all_pe, dim=0)
                u_ppe = torch.cat(all_ppe, dim=0)

                indices = torch.tensor(
                    [p_to_idx[p] for p in initial_prompts], device=device
                )
                pe_init = u_pe[indices]
                ppe_init = u_ppe[indices]
        else:
            print("Encoding empty prompts...")
            with torch.no_grad():
                prompt = ""
                pe, ppe, _ = pipe.encode_prompt(
                    prompt=[prompt], prompt_2=None, device=device
                )
                pe_init = pe.repeat(num_channels, 1, 1)
                ppe_init = ppe.repeat(num_channels, 1)

        self.register_buffer("pe_anchor", pe_init.clone())
        self.register_buffer("ppe_anchor", ppe_init.clone())

        self.pe_lora_A = nn.Parameter(torch.zeros(num_channels, 512, rank, device=device))
        self.pe_lora_B = nn.Parameter(torch.zeros(num_channels, rank, 4096, device=device))

        self.ppe_delta = nn.Parameter(torch.zeros_like(ppe_init) * 0.1)
        self.pe_logvar = nn.Parameter(torch.full((num_channels, 512, 1), -12.0, device=device))
        self.ppe_logvar = nn.Parameter(torch.full_like(ppe_init, -12.0))

    def forward(self, channel_indices):
        A = self.pe_lora_A[channel_indices]
        B = self.pe_lora_B[channel_indices]
        delta = torch.bmm(A, B)
        pe_mu = self.pe_anchor[channel_indices] + delta
        pe_std = torch.exp(0.5 * self.pe_logvar[channel_indices])
        pe = pe_mu + torch.randn_like(pe_std) * pe_std

        ppe_mu = self.ppe_anchor[channel_indices] + self.ppe_delta[channel_indices]
        ppe_std = torch.exp(0.5 * self.ppe_logvar[channel_indices])
        ppe = ppe_mu + torch.randn_like(ppe_std) * ppe_std
        return (
            pe,
            ppe,
            self.pe_anchor[channel_indices],
            self.ppe_anchor[channel_indices],
        )

    def reg_loss(self, channel_indices):
        return self.pe_lora_A[channel_indices].pow(2).mean() + \
               self.pe_lora_B[channel_indices].pow(2).mean() + \
               self.ppe_delta[channel_indices].pow(2).mean()


def differentiable_flux_generate(
    pipe,
    prompt_embeds,
    pooled_prompt_embeds,
    num_steps=4,
    guidance_scale=None,
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

    if guidance_scale is not None:
        guidance_vec = torch.full(
            (batch_size,), guidance_scale, device=device, dtype=torch.bfloat16
        )
    else:
        guidance_vec = None

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

    initial_prompts = discover_initial_prompts(config, pipe, backbone, device)

    U = create_matrix(config.matrix.type, model_bundle.num_channels, device).to(
        torch.float32
    )
    mod_head = create_modified_head(base_model, config.model.name, U=U)

    prompt_bank = VariationalPromptBank(
        model_bundle.num_channels, pipe, device, initial_prompts
    )

    opt_prompts = torch.optim.AdamW(
        prompt_bank.parameters(), lr=config.training.lr_reg, weight_decay=0.0
    )
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

    CYCLE_LENGTH = config.training.U_steps + config.training.prompt_steps

    set_seeds(config.seed)

    for step in pbar:
        target_channels = torch.randint(
            0, model_bundle.num_channels, (config.training.batch_size,), device=device
        )

        is_U_phase = (step % CYCLE_LENGTH) < config.training.U_steps
        if step < config.training.warmup_steps:
            is_U_phase = True


        with torch.autocast("cuda", dtype=torch.bfloat16):
            pe, ppe, _, _ = prompt_bank(target_channels)
            generated_imgs = differentiable_flux_generate(
                pipe,
                pe,
                ppe,
                num_steps=config.generative_model.gen_steps,
                guidance_scale=config.generative_model.guidance_scale,
                device=device,
            )

        imgs_in = F.interpolate(generated_imgs, (224, 224), mode="bilinear")
        feats = backbone((imgs_in.float() - mean) / std).to(torch.float32)

        if is_U_phase:
            purity_scores, _, _ = epic_purity(feats.detach(), U(), target_channels)
            loss_purity = -config.training.lambda_purity * purity_scores.mean()
            opt_U.zero_grad(set_to_none=True)
            loss_purity.backward()
            opt_U.step()
        else:
            purity_scores, _, _ = epic_purity(feats, U().detach(), target_channels)
            loss_purity = -config.training.lambda_purity * purity_scores.mean()
            if config.training.lambda_reg != 0:
                loss_reg = config.training.lambda_reg * prompt_bank.reg_loss(
                    target_channels
                )
            else:
                loss_reg = 0.0

            total_loss = loss_purity + loss_reg

            opt_prompts.zero_grad(set_to_none=True)
            opt_U.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(prompt_bank.parameters(), 1.0)
            opt_prompts.step()

        torch.cuda.empty_cache()
        purity_history.append(purity_scores.mean().item())

        if step % 200 == 0 or (step + 1) == config.training.steps:
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

            with torch.no_grad():
                img_v_norm = (img_v_cuda - mean) / std
                v_feats_raw = backbone(img_v_norm)
                v_feats_rot = pixelwise_multiply(
                    v_feats_raw.to(torch.float32),
                    U().to(torch.float32),
                )

                hm_orig = get_heatmap(
                    backbone, mod_head, img_v_norm, batch_size=1, device=device
                )

            top_ch = topk_active_channels(
                backbone, mod_head, img_v_cuda[0], k=4, device=device
            )
            fig, axes = plt.subplots(4, 6, figsize=(24, 16))
            fig_overlay, axes_overlay = plt.subplots(4, 6, figsize=(24, 16))

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

                hm_c = hm_orig[0, ch]
                hm_c_norm = hm_c / (np.max(hm_c) + 1e-8)

                axes_overlay[row, 0].imshow(np.array(orig_pil))
                axes_overlay[row, 0].imshow(hm_c_norm, cmap="jet", alpha=0.6)
                axes_overlay[row, 0].set_ylabel(
                    f"Channel {ch}", fontsize=14, fontweight="bold"
                )
                axes_overlay[row, 0].set_xticks([])
                axes_overlay[row, 0].set_yticks([])

                ch_tensor = torch.tensor([ch] * 5, device=device)

                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        pe_vis, ppe_vis, _, _ = prompt_bank(ch_tensor)

                        ex_imgs_t = differentiable_flux_generate(
                            pipe,
                            pe_vis,
                            ppe_vis,
                            num_steps=config.generative_model.gen_steps,
                            guidance_scale=config.generative_model.guidance_scale,
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
                        proto_in_norm = (proto_in - mean) / std
                        proto_feats = backbone(proto_in_norm)
                        proto_rot = pixelwise_multiply(
                            proto_feats.to(torch.float32),
                            U().to(torch.float32),
                        )

                        hm_proto = get_heatmap(
                            backbone,
                            mod_head,
                            proto_in_norm,
                            batch_size=1,
                            device=device,
                        )

                    px1, py1, px2, py2 = compute_activation_bbox(
                        proto_rot[0, ch].unsqueeze(0), (224, 224)
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

                    hm_p_c = hm_proto[0, ch]
                    hm_p_c_tensor = torch.from_numpy(hm_p_c).unsqueeze(0).unsqueeze(0)
                    hm_p_c_resized = (
                        F.interpolate(
                            hm_p_c_tensor,
                            size=(224, 224),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze()
                        .numpy()
                    )

                    hm_p_c_norm = hm_p_c_resized / (np.max(hm_p_c_resized) + 1e-8)

                    axes_overlay[row, col + 1].imshow(np.array(proto_pil))
                    axes_overlay[row, col + 1].imshow(
                        hm_p_c_norm, cmap="jet", alpha=0.6
                    )
                    axes_overlay[row, col + 1].axis("off")

            fig.tight_layout()
            fig.savefig(
                os.path.join(grid_dir, f"step_{step + 1:04d}.jpg"), bbox_inches="tight"
            )
            plt.close(fig)

            fig_overlay.tight_layout()
            fig_overlay.savefig(
                os.path.join(grid_dir, f"step_{step + 1:04d}_overlay.jpg"),
                bbox_inches="tight",
            )
            plt.close(fig_overlay)

            gc.collect()
            torch.cuda.empty_cache()

        if step % 5 == 0:
            pbar.set_postfix({"purity": f"{purity_scores.mean().item():.4f}"})

        if step % 200 == 0:
            U.save_state(config.output_path, f"{config.matrix.type}.pt")
            torch.save(
                prompt_bank.state_dict(),
                os.path.join(config.output_path, "learned_prompts.pt"),
            )

    U.save_state(config.output_path, f"{config.matrix.type}.pt")
    torch.save(
        prompt_bank.state_dict(), os.path.join(config.output_path, "learned_prompts.pt")
    )
