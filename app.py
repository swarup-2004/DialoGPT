import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="DialoGPT Chatbot", layout="centered")
st.title("💬 Chat with DialoGPT")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

# Initialize session state for chat history
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "step" not in st.session_state:
    st.session_state.step = 0

# User input box
user_input = st.chat_input("Say something...")

if user_input:
    st.chat_message("user").markdown(user_input)

    # Encode user input + EOS token
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Append to chat history if exists
    bot_input_ids = torch.cat(
        [st.session_state.chat_history_ids, new_input_ids], dim=-1
    ) if st.session_state.chat_history_ids is not None else new_input_ids

    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the response
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True
    )

    # Display assistant's message
    st.chat_message("assistant").markdown(response)

    # Update session state
    st.session_state.chat_history_ids = chat_history_ids
    st.session_state.step += 1
