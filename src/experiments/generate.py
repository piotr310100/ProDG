import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import Normalize, ToPILImage, ToTensor
from torchvision.transforms.functional import resize
from tqdm import tqdm

from data import create_indexed_dataloader
from matrix import create_matrix
from models import create_backbone_model
from prototypes import (
    compute_activation_bbox,
    generate_prototypes,
    pixelwise_multiply,
    purity_argmax,
    unnormalize,
)

norm_transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = ToTensor()


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_purity_wrapper(
    x0_image_tensor, feature_model, U, target_channel, mean, std
):
    img_01 = (x0_image_tensor / 2 + 0.5).clamp(0, 1)

    img_resized = F.interpolate(
        img_01, size=(224, 224), mode="bilinear", align_corners=False
    )
    img_norm = (img_resized - mean) / std

    features = feature_model(img_norm)
    rotated_features = pixelwise_multiply(features, U)

    return -purity_argmax(rotated_features, target_channel)


def draw_box_and_get_purity(img_pil, feature_model, U, channel_idx, device):
    img_resized_pil = img_pil.resize((224, 224), Image.Resampling.BILINEAR)
    img_tensor = to_tensor(img_resized_pil).to(device)
    img_norm = norm_transform(img_tensor).unsqueeze(0)

    with torch.no_grad():
        features = feature_model(img_norm)
        if U is not None:
            features = pixelwise_multiply(features, U)

        purity_val = purity_argmax(features, channel_idx).item()
        channel_activation = features[:, channel_idx]

    x_start, y_start, x_end, y_end = compute_activation_bbox(
        channel_activation, (224, 224)
    )

    orig_w, orig_h = img_pil.size
    scale_x = orig_w / 224.0
    scale_y = orig_h / 224.0

    rect = [
        int(x_start.item() * scale_x),
        int(y_start.item() * scale_y),
        int(x_end.item() * scale_x) - 1,
        int(y_end.item() * scale_y) - 1,
    ]

    draw = ImageDraw.Draw(img_pil)
    draw.rectangle(rect, outline=(255, 255, 0), width=3)

    return img_pil, purity_val


def get_real_prototypes_with_scores(
    feature_model,
    positive_prototypes,
    orig_dataloader,
    channels,
    U=None,
    device="cpu",
    transform=unnormalize,
):
    img_transform = lambda P, Z: Z
    result = {}

    for c in channels:
        result[c] = []
        prototypes_idx = positive_prototypes[c]

        batch = torch.stack(
            [orig_dataloader.dataset[idx][0] for idx in prototypes_idx]
        ).to(device=device, non_blocking=True)

        feature_map = feature_model(batch)
        if U is not None:
            feature_map = pixelwise_multiply(feature_map, U)

        purities = purity_argmax(feature_map, c)
        channel_activation = feature_map[:, c]

        x_start, y_start, x_end, y_end = compute_activation_bbox(
            channel_activation, batch.size()[2:]
        )

        R_k = resize(channel_activation.unsqueeze(1), (batch.size(-2), batch.size(-1)))
        S_k = torch.nn.functional.relu(R_k)
        P_k = S_k / torch.max(S_k).clamp_min(1e-8)
        Z = transform(batch)

        for i in range(batch.size(0)):
            prototype_image = img_transform(P_k[i], Z[i])
            prototype_image = ToPILImage()(prototype_image.cpu())

            draw = ImageDraw.Draw(prototype_image)
            draw.rectangle(
                [
                    x_start[i].item(),
                    y_start[i].item(),
                    x_end[i].item() - 1,
                    y_end[i].item() - 1,
                ],
                outline=(255, 255, 0),
                width=2,
            )
            result[c].append((prototype_image, purities[i].item()))

    return result


def create_comparison_grid(real_data, gen_data, channel_idx):
    display_h = 256
    text_h = 30

    def prepare_row(data_list):
        resized_imgs = []
        scores = []
        for img, score in data_list:
            aspect = img.width / img.height
            new_w = int(display_h * aspect)
            resized_imgs.append(
                img.resize((new_w, display_h), Image.Resampling.LANCZOS)
            )
            scores.append(f"{score:.4f}")
        return resized_imgs, scores

    real_imgs, real_scores = prepare_row(real_data)
    gen_imgs, gen_scores = prepare_row(gen_data)

    row1_w = sum(img.width for img in real_imgs) + (len(real_imgs) - 1) * 10
    row2_w = sum(img.width for img in gen_imgs) + (len(gen_imgs) - 1) * 10

    total_w = max(row1_w, row2_w, 400) + 40
    total_h = 100 + (display_h + text_h) + 60 + (display_h + text_h) + 20

    combined = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(combined)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
        font_lg = ImageFont.truetype("arial.ttf", 40)
        font_score = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        font_lg = ImageFont.load_default()
        font_score = ImageFont.load_default()

    draw.text((20, 10), f"Channel {channel_idx} Analysis", fill="black", font=font_lg)

    y_start_real = 60
    draw.text((20, y_start_real), "Real Prototypes (Dataset)", fill="black", font=font)

    curr_x = 20
    y_img_real = y_start_real + 40

    for i, img in enumerate(real_imgs):
        combined.paste(img, (curr_x, y_img_real))
        score_text = f"Purity: {real_scores[i]}"
        bbox = draw.textbbox((0, 0), score_text, font=font_score)
        text_w = bbox[2] - bbox[0]
        text_x = curr_x + (img.width - text_w) // 2
        draw.text(
            (text_x, y_img_real + display_h + 5),
            score_text,
            fill="black",
            font=font_score,
        )
        curr_x += img.width + 10

    y_start_gen = y_img_real + display_h + text_h + 40
    draw.text(
        (20, y_start_gen), "Generated Prototypes (EPIC Guided)", fill="black", font=font
    )

    curr_x = 20
    y_img_gen = y_start_gen + 40

    for i, img in enumerate(gen_imgs):
        combined.paste(img, (curr_x, y_img_gen))
        score_text = f"Purity: {gen_scores[i]}"
        bbox = draw.textbbox((0, 0), score_text, font=font_score)
        text_w = bbox[2] - bbox[0]
        text_x = curr_x + (img.width - text_w) // 2
        draw.text(
            (text_x, y_img_gen + display_h + 5),
            score_text,
            fill="black",
            font=font_score,
        )
        curr_x += img.width + 10

    return combined


def generate_and_compare(config: DictConfig):
    set_seeds(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = os.path.join(config.output_path, "Comparison_Prototypes")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading EPIC Backbone: {config.model.name}")
    _, feature_model, transform, num_channels = create_backbone_model(
        config.model.name,
        device,
        config.model.custom_weights_path,
        config.model.num_classes,
    )
    feature_model = feature_model.to(device).eval().requires_grad_(False).float()

    disentanglement_matrix = create_matrix(config.matrix.type, num_channels, device)
    disentanglement_matrix.load_state(
        os.path.join(config.output_path, f"{config.matrix.type}.pt"),
        map_location=device,
    )
    U_matrix = disentanglement_matrix().detach().float()

    num_real_protos = config.visualization.num_prototypes
    print(f"Calculating {num_real_protos} Real Prototypes per channel...")

    dataloader_train = create_indexed_dataloader(
        config.dataset.datapath_train,
        transform,
        config.dataloader.batch_size,
        config.dataloader.num_workers,
        config.dataloader.pin_memory,
        shuffle=False,
    )

    positive_prototypes_indices = generate_prototypes(
        feature_model,
        dataloader_train,
        num_channels,
        N=num_real_protos,
        device=device,
        U=U_matrix,
    )

    if config.generation.target_channels:
        target_channels = config.generation.target_channels
    else:
        target_channels = torch.randint(
            0, num_channels, (config.visualization.num_prototypes,)
        ).tolist()

    real_prototypes_dict = get_real_prototypes_with_scores(
        feature_model,
        positive_prototypes_indices,
        dataloader_train,
        channels=target_channels,
        U=U_matrix,
        device=device,
    )
    del dataloader_train
    torch.cuda.empty_cache()

    print(f"Loading SD: {config.generation.model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        config.generation.model_id, torch_dtype=torch.float16, use_safetensors=True
    ).to(device)

    pipe.vae.to(dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    prompt = config.generation.prompt
    neg_prompt = config.generation.negative_prompt
    lambda_scale = config.generation.purity_scale
    num_steps = config.generation.num_steps
    num_gen_protos = config.visualization.num_prototypes

    print(f"Prompt: {prompt}")

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt, device, 1, True, neg_prompt
    )[:2]
    text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])

    for channel_idx in target_channels:
        generated_data = []

        for gen_idx in range(num_gen_protos):
            pipe.scheduler.set_timesteps(num_steps, device=device)
            timesteps = pipe.scheduler.timesteps

            latents = (
                torch.randn(
                    (1, pipe.unet.config.in_channels, 512 // 8, 512 // 8),
                    device=device,
                    dtype=torch.float16,
                )
                * pipe.scheduler.init_noise_sigma
            )

            iterator = tqdm(
                timesteps,
                desc=f"Ch {channel_idx} | Gen {gen_idx + 1}/{num_gen_protos}",
                leave=False,
            )

            for i, t in enumerate(iterator):
                latents = latents.detach().requires_grad_(True)

                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = pipe.scheduler.scale_model_input(
                    latent_model_input, t
                )

                with torch.no_grad():
                    noise_pred = pipe.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    ).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + config.generation.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                shift = 0

                if lambda_scale > 0:
                    step_output = pipe.scheduler.step(noise_pred, t, latents)
                    x0_latents = step_output.pred_original_sample

                    scaled_x0 = 1 / pipe.vae.config.scaling_factor * x0_latents.float()
                    x0_image = pipe.vae.decode(scaled_x0).sample

                    loss = compute_purity_wrapper(
                        x0_image, feature_model, U_matrix, channel_idx, mean, std
                    )

                    grad = torch.autograd.grad(loss, latents)[0]

                    g_norm = torch.linalg.norm(grad)
                    g_normalized = grad / (g_norm + 1e-6)

                    shift = (-lambda_scale * g_normalized).to(latents.dtype)

                    iterator.set_postfix({"loss": f"{loss.item():.4f}"})

                with torch.no_grad():
                    step_output = pipe.scheduler.step(noise_pred, t, latents)
                    prev_latents = step_output.prev_sample
                    latents = prev_latents + shift

            with torch.no_grad():
                latents = 1 / pipe.vae.config.scaling_factor * latents
                image = pipe.vae.decode(latents.float()).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                image = (image * 255).round().astype("uint8")[0]
                gen_pil = Image.fromarray(image)

                gen_pil, score = draw_box_and_get_purity(
                    gen_pil, feature_model, U_matrix, channel_idx, device
                )
                generated_data.append((gen_pil, score))

        real_data = []
        if channel_idx in real_prototypes_dict:
            real_data = real_prototypes_dict[channel_idx]

        while len(real_data) < num_real_protos:
            real_data.append((Image.new("RGB", (224, 224), (0, 0, 0)), 0.0))

        combined_img = create_comparison_grid(real_data, generated_data, channel_idx)
        save_path = os.path.join(output_dir, f"comparison_ch{channel_idx}.jpg")
        combined_img.save(save_path)
        print("Saved comparison:", save_path)

    print("Comparison complete.")
