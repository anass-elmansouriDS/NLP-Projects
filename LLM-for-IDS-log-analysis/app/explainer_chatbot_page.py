import streamlit as st
import torch
import time
from llm_helpers import get_final_response

def explainer_chatbot() :
    """
    Sets up the explainer and cybersecurity chatbot streamlit page.
    """
    if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "user", "text": "Explain the classifier's decision"},{"role": "assistant", "text": st.session_state.response}]
    st.subheader("Cybersecurity chatbot")
    col5,col6=st.columns(2)
    with col5 :
        if st.button("New Chat") :
            st.session_state.messages=[] 
    with col6 :
        if st.button('Back to the network classifier'):
            st.session_state.active_tab="Network Traffic Classifier"
            st.rerun()
    
    # Function to simulate chatbot response
    def get_chatbot_response(user_message):
        # You can customize this to have a more intelligent response
        return get_final_response(user_message,st.session_state.chain,conversation=True)

    #show messages history
    for message in st.session_state.messages :
        with st.chat_message(message["role"]):
            st.markdown(message["text"])            
    
    # User input for chatbot
    if prompt := st.chat_input("Talk to cyberexplainer :") :
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "text": prompt})
        
        # Get chatbot's response
        chatbot_response = get_chatbot_response(prompt)
        #show chatbot response 
        with st.chat_message("assistant"):
            def stream_message() :
                for char in chatbot_response.split(" ") :
                    yield char + " "
                    time.sleep(0.044)
            st.write_stream(stream_message)
            
        # Add chatbot response to chat history
        st.session_state.messages.append({"role": "assistant", "text": chatbot_response})
        torch.cuda.empty_cache()
