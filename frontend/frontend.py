
import streamlit as st
import requests

st.title("Chatbot")

if st.button("Call API/get_all"):
    response = requests.get("http://api:8000/get_all")
    st.write(response.json()["message"])

def get_chat():

    st.session_state["responses"] = []
    st.session_state["responses"].add()

ui_style = """
<style>
    .stMainBlockContainer{
        background-color: #00ffb22e;
    }
</style>
"""
st.markdown(ui_style, unsafe_allow_html=True)


def chat_show():
    
    if "responses" not in st.session_state:
        st.session_state["responses"] = []

    num_responses = len(st.session_state["responses"])
    if num_responses > 0:
        for i in st.session_state["responses"]:
            print(i)

def chat_input():
    pass


chat_show()
chat_input()









