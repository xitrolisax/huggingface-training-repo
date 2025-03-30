from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

output = generator(
    "Hello, I am",
    max_length=50,  
    num_return_sequences=1, 
    temperature=0.9, 
    top_p=0.95,  
    repetition_penalty=1.2  
)

print(output)
