
from transformers import AutoTokenizer, pipeline

def tokenization():
    extractor = pipeline("feature-extraction", model="bert-base-uncased")
    result = extractor("Hugging Face is amazing!")
    print(result)

def word_piece_tokenization():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "Hugging Face is amazing!"
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(tokens)
    decoded_text = tokenizer.decode(ids)
    print(decoded_text)

def bpe_tokenization():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = "Hugging Face is amazing!"
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    decoded_text = tokenizer.decode(ids)
    print(tokens)  
    print(decoded_text)

bpe_tokenization()