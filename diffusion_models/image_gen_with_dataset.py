import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import get_peft_model, LoraConfig
from tqdm import tqdm

# ============ НАСТРОЙКИ ============
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float32

dataset_path = "/Users/olgapopova/Desktop/my_dataset"  # путь к UI-дизайнам
image_size = 512
batch_size = 1
epochs = 5
lr = 1e-4
output_dir = "./lora_ui_model"

# ============ ПРЕПРОЦЕССИНГ ============
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = ImageFolder(dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ============ PIPELINE И КОМПОНЕНТЫ ============
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
vae = pipe.vae.to(device)
unet = pipe.unet
text_encoder = pipe.text_encoder.to(device)
tokenizer = pipe.tokenizer
scheduler = pipe.scheduler

unet.to(device, dtype=dtype)

# ============ LoRA КОНФИГ ============
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "proj_in", "proj_out"],
    lora_dropout=0.1,
    bias="none"
)
unet = get_peft_model(unet, lora_config)
unet.train()

# ============ ОПТИМАЙЗЕР ============
optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# ============ ТРЕНИРОВКА ============
print("\n🚀 Обучение началось!")
for epoch in range(epochs):
    total_loss = 0
    for step, (images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
        images = images.to(device, dtype=dtype)

        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * 0.18215

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        prompt = ["a website ui design"] * images.shape[0]
        inputs = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt").to(device)
        encoder_hidden_states = text_encoder(**inputs).last_hidden_state

        optimizer.zero_grad()
        model_pred = unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample

        loss = loss_fn(model_pred, noise)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"\n📉 Loss за эпоху {epoch+1}: {avg_loss:.4f}")

# ============ СОХРАНЕНИЕ ============
os.makedirs(output_dir, exist_ok=True)
unet.save_pretrained(output_dir)
print(f"\n✅ LoRA сохранена в {output_dir}")
