import torch
from diffusers import StableDiffusionPipeline

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

prompt = "cute design"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("ui_mockup.png")
image.show()
