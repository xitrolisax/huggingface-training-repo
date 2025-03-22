import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipeline.to(device)
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

image_path = "/Users/olgapopova/Desktop/hugging face nlp - 1/input_image.png"
image = load_image(image_path)

prompt = "A futuristic cyberpunk character with a dynamic pose, ultra-detailed, 4K"
generated_image = pipeline(prompt, image=image, num_inference_steps=80, guidance_scale=9).images[0]

generated_image.save("output_image.png")

generated_image.show()
