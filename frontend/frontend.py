
import streamlit as st
import requests



def chat_show():
    
    if "responses" not in st.session_state:
        # st.write("responses was empty")
        st.session_state["responses"] = []
    
    num_responses = len(st.session_state["responses"])
    if num_responses > 0:
        chat_field.empty()
        chat_field.write(num_responses)
        text = ""

        for role, content in st.session_state["responses"]:
            text += f"**{role}:**  {content}<br>"

        chat_field.markdown(text, unsafe_allow_html=True)


def chat_input():

    with st.form(key="input_form"):
        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.text_input(label="Type query here:")

        with col2:
            submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        # st.write(f"You entered: {user_input}")

        data = {"query": user_input, "history": st.session_state["responses"]}

        response = requests.post("http://api:8000/api/query", json=data)
        
        response_data = response.json()

        st.write(response_data["history"])
        st.session_state["responses"] = response_data["history"]

        chat_show()



ui_style = """
<style>
    .stMainBlockContainer{
        // background-color: #00ffb22e;
        // background-color: #f1f1f1;
    }

    .stForm{
        // background-color: blue;
    }

    .stColumn{
        display: flex;
        flex-direction: row;
        align-items: flex-end;
        // background-color: red;
    }

    .stColumn div{
        flex-grow: 1;
    }

</style>
"""
st.markdown(ui_style, unsafe_allow_html=True)


st.title("Chatbot")


chat_field = st.empty()

chat_show()

chat_input()

if st.button("update"):
    pass














