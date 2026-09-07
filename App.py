import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_model()

st.title("Resume Summarizer")

text = st.text_area("Paste your resume text here:")

if st.button("Summarize"):
    if text:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            num_beams=4,
            early_stopping=True
        )

        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        st.subheader("Summary")
        st.write(summary)
    else:
        st.warning("Please enter resume text.")