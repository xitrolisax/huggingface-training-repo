import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
pipe.to(device)

pipe.unet = PeftModel.from_pretrained(pipe.unet, "./lora_ui_model")
pipe.unet.to(device, dtype=torch_dtype)

def generate_ui(prompt, steps=40, scale=8.0, out_path="ui_generated.png"):
    generator = torch.manual_seed(42)
    image = pipe(prompt, num_inference_steps=steps, guidance_scale=scale, generator=generator).images[0]
    image.save(out_path)
    return image

prompt = "minimalist website design"
image = generate_ui(prompt)
image.show()
