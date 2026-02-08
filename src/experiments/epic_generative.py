import gc
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    flat = rotated.view(B, C, -1)
    max_vals, _ = flat.max(dim=2)
    l2 = torch.norm(flat, dim=2).clamp_min(1e-6)
    return (
        max_vals[torch.arange(B), target_channels]
        / l2[torch.arange(B), target_channels]
    )


def decode_latents(pipeline, latents, height, width):
    lv = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    lv = (lv / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    return pipeline.vae.decode(
        lv.to(device=latents.device, dtype=pipeline.vae.dtype)
    ).sample


def generate_prototype(
    pipe,
    backbone,
    modified_head,
    U_tensor,
    device,
    mean,
    std,
    target_class_idx=None,
    channel_idx=0,
    n=1,
    num_steps=4,
    purity_guidance_scale=3.5,
    logit_scale=2.0,
    seed=None,
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    height, width = 512, 512
    # prompt = ""
    prompt = "a professional high-quality photo of a bird, sharp focus"
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=[prompt] * n, prompt_2=None, device=device
        )

    latents, latent_ids = pipe.prepare_latents(
        batch_size=n,
        num_channels_latents=pipe.transformer.config.in_channels // 4,
        height=height,
        width=width,
        dtype=pipe.dtype,
        device=device,
        generator=torch.Generator(device=device).manual_seed(seed) if seed else None,
    )

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
        guidance_vec = torch.full((n,), 3.5, device=device, dtype=pipe.dtype)
    else:
        pipe.scheduler.set_timesteps(num_steps, device=device)
        guidance_vec = None

    for i, t in enumerate(pipe.scheduler.timesteps):
        progress = i / len(pipe.scheduler.timesteps)

        is_guidance_step = progress < 0.85
        if is_guidance_step:
            latents = latents.detach().requires_grad_(True)

        with torch.no_grad():
            noise_pred = pipe.transformer(
                hidden_states=latents,
                timestep=t.expand(n).to(device) / 1000.0,
                guidance=guidance_vec,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]

        step_out = pipe.scheduler.step(noise_pred, t, latents, return_dict=True)

        if is_guidance_step:
            t_norm = (t / 1000.0).to(latents.dtype)
            x0_packed = latents - t_norm * noise_pred

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                img = decode_latents(pipe, x0_packed, height, width)
                img_01 = (img.float() / 2 + 0.5).clamp(0, 1)
                x_in = F.interpolate(img_01, (224, 224), mode="bilinear")

                feats = backbone((x_in - mean) / std)
                rotated = pixelwise_multiply(
                    feats.to(torch.float32), U_tensor.to(torch.float32)
                )
                flat = rotated.view(n, rotated.shape[1], -1)
                max_vals, _ = flat.max(dim=2)
                l2 = torch.norm(flat, dim=2).clamp_min(1e-6)
                purity = (
                    max_vals[torch.arange(n), channel_idx]
                    / l2[torch.arange(n), channel_idx]
                )

                loss = -purity.mean()
                if target_class_idx is not None:
                    loss -= (
                        logit_scale * modified_head(feats)[:, target_class_idx].mean()
                    )

            grad = torch.autograd.grad(loss, latents)[0]

            with torch.no_grad():
                norm = grad.norm().clamp_min(1e-6)
                shift = (grad / norm) * purity_guidance_scale * (1.0 - progress)
                latents = step_out.prev_sample.detach() - shift
        else:
            with torch.no_grad():
                latents = step_out.prev_sample.detach()

        torch.cuda.empty_cache()

    with torch.no_grad():
        final_img = decode_latents(pipe, latents, height, width)
    return (final_img.float() / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()


def run_epic_generative(config: DictConfig):
    set_seeds(config.seed)
    device = torch.device("cuda")

    model_bundle = create_backbone_model(
        config.model.name,
        device,
        config.model.custom_weights_path,
        config.model.num_classes,
    )
    backbone = model_bundle.backbone.to(torch.float32).eval()
    base_model = model_bundle.base_model.to(torch.float32).eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    m_3d, s_3d = mean.squeeze().view(3, 1, 1).cpu(), std.squeeze().view(3, 1, 1).cpu()

    pipe = FluxPipeline.from_pretrained(
        config.generative_model.model_id, torch_dtype=torch.bfloat16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.to(torch.float32)

    U = create_matrix(config.matrix.type, model_bundle.num_channels, device).to(
        torch.float32
    )
    optimizer = torch.optim.Adam(U.parameters(), lr=config.training.lr)

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

    pbar = tqdm(range(config.training.steps), desc="EPIC Training")
    for step in pbar:
        B = config.training.batch_size
        target_channels = torch.randint(
            0, model_bundle.num_channels, (B,), device=device
        )
        mod_head = create_modified_head(base_model, config.model.name, U=U)

        feats_list = []
        for i in range(B):
            img_np = generate_prototype(
                pipe,
                backbone,
                mod_head,
                U(),
                device,
                mean,
                std,
                channel_idx=target_channels[i],
                num_steps=config.training.gen_steps,
                purity_guidance_scale=config.training.purity_guidance_scale,
            )
            with torch.no_grad():
                img_t = (
                    torch.from_numpy(img_np)
                    .permute(0, 3, 1, 2)
                    .to(device, torch.float32)
                )
                feats_list.append(
                    backbone((F.interpolate(img_t, (224, 224)) - mean) / std)
                )

        cached_feats = torch.cat(feats_list, dim=0).detach()
        for _ in range(config.training.get("u_steps_per_gen", 1)):
            optimizer.zero_grad(set_to_none=True)
            purity_train = epic_purity(cached_feats, U(), target_channels)
            (-purity_train.mean()).backward()
            optimizer.step()
        purity_history.append(purity_train.mean().item())

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
            with torch.no_grad():
                v_feats_raw = backbone((img_v_cuda - mean) / std)
                v_feats_rot = pixelwise_multiply(
                    v_feats_raw.to(torch.float32),
                    U().to(torch.float32),
                )
                predicted_class = mod_head(v_feats_raw).argmax(dim=1).item()

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

                ex_imgs = generate_prototype(
                    pipe,
                    backbone,
                    mod_head,
                    U(),
                    device,
                    mean,
                    std,
                    target_class_idx=predicted_class,
                    channel_idx=ch,
                    n=5,
                    num_steps=config.training.gen_steps,
                    purity_guidance_scale=config.training.purity_guidance_scale,
                )

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
                        proto_rot[0, ch].unsqueeze(0), (512, 512)
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
            pbar.set_postfix({"purity": f"{purity_train.mean().item():.4f}"})

    U.save_state(config.output_path, f"{config.matrix.type}.pt")
