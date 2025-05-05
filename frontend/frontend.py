
import streamlit as st
import requests



def chat_show():
    
    if "responses" not in st.session_state:
        st.session_state["responses"] = []
    
    num_responses = len(st.session_state["responses"])
    if num_responses > 0:
        chat_field.empty()
        chat_field.write(num_responses)
        text = ""

        for role, content in st.session_state["responses"]:
            text += f"**{role}:**  {content}<br><br>"

        chat_field.markdown(text, unsafe_allow_html=True)


def chat_input():

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
        st.session_state.user_input_internal = ""

    def clear_input():
        st.session_state.user_input_internal = st.session_state.user_input
        st.session_state.user_input = ""
        

    with st.form(key="input_form"):
        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.text_input(label="Type query here:", key="user_input")

        with col2:
            submit_button = st.form_submit_button(label="Submit", on_click=clear_input)

    if submit_button:
        # st.write(f"input: {st.session_state.user_input}")
        # st.write(f"internal: {st.session_state.user_input_internal}")

        data = {
            "query": st.session_state.user_input_internal, 
            "history": st.session_state["responses"]
        }

        # response = requests.post("http://api:8000/api/query", json=data)
        response = requests.post("http://api:8000/api/query_with_context", json=data)
        
        response_data = response.json()

        st.write(response_data["history"])
        st.session_state["responses"] = response_data["history"]

        chat_show()


def items_show():
    

    fetch_response = requests.get("http://api:8000/api/get_all")
    fetch_response_data = fetch_response.json()

    st.write(fetch_response_data)


    for item in fetch_response_data["message"]:
        st.subheader(f"item: {item['uuid']}")
        st.text(f"Properties: {item['properties']}")

        if st.button(f"Delete {item['uuid']}", key=f"delete_id_{item['uuid']}"):

            data = {"Collection": "Symptoms", "uuid": item['uuid']}

            delete_response = requests.post("http://api:8000/api/delete_object", json=data)
            delete_response_data = delete_response.json()

            st.write(delete_response_data)
            




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

items_show()

if st.button("update"):
    # st.session_state.user_input = ""
    pass

if st.button("reset symptoms"):
    response = requests.get("http://api:8000/api/create_collection_symptoms")
    print(response)














