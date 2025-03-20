from transformers import pipeline

def sentiment_analysis():
    classifier = pipeline("sentiment-analysis")
    result_first = classifier("Hugging Face.")
    result_second = classifier("Hugging Face is the best NLP!")
    result_third = classifier("Hugging Face is the worst NLP!")
    print(result_first)
    print(result_second)
    print(result_third)

def text_classification():
    candidate_labels = ["positive", "negative", "neutral"]
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result_first = classifier("Hugging Face is revolutionizing NLP!", candidate_labels)
    result_second = classifier("Hugging Face is the best NLP!", candidate_labels)
    result_third = classifier("Hugging Face is the worst NLP!", candidate_labels)
    print(result_first)
    print(result_second)
    print(result_third)

sentiment_analysis()
text_classification()