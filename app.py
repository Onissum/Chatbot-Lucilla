import streamlit as st
from transformers import pipeline, MarianMTModel, MarianTokenizer, AutoModelForCausalLM, AutoTokenizer

# Inizializza il traduttore italiano-inglese
tokenizer_it_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-en")
model_it_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-it-en")

# Inizializza il traduttore inglese-italiano
tokenizer_en_it = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")
model_en_it = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-it")

# Carica il modello fairseq-dense-2.7B
model_name = "KoboldAI/fairseq-dense-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Creazione della pipeline di generazione del testo
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Funzione per la traduzione
def translate(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interfaccia utente Streamlit per il chatbot
st.title("Chatbot Lucilla")

user_input = st.text_input("Scrivi qui per parlare con Lucilla:")

if st.button("Invia"):
    input_in_english = translate(user_input, model_it_en, tokenizer_it_en)
    responses = chatbot(input_in_english, max_length=50, num_return_sequences=1, temperature=0.9, do_sample=True)
    response_in_english = responses[0]['generated_text']
    response_in_italian = translate(response_in_english, model_en_it, tokenizer_en_it)
    st.text_area("Risposta:", value=response_in_italian, height=100)

# Slider esempio
st.write("Esempio Slider:")
x = st.slider('Select a value')
st.write(x, 'squared is', x * x)
