import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("imdb")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True)
test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=8)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 3 
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_dataloader, leave=True)
    
    for batch in loop:
        inputs = {key: val.to(device) for key, val in batch.items()}
        labels = inputs.pop("label")  

        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

print("✅ Обучение завершено!")

model.save_pretrained("bert-imdb-model")
tokenizer.save_pretrained("bert-imdb-model")

classifier = pipeline("text-classification", model="bert-imdb-model", tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

texts = [
    "This movie was an absolute masterpiece! The performances were stunning.",
    "I hated every second of this film. The acting was terrible."
]

results = classifier(texts)
print("📌 Тестовые результаты:", results)
