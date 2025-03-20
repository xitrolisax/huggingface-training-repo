import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = classifier(text)[0]  
    return f"🔹 **Tone:** {result['label']} (🔢 Approx: {result['score']:.2f})"

interface = gr.Interface(
    fn=analyze_sentiment, 
    inputs=gr.Textbox(lines=3, placeholder="Enter the text..."), 
    outputs=gr.Textbox(),
    title="Tone analysis",
    description="Enter your text to analyze its sentiment"
)

interface.launch()
