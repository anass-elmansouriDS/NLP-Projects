import streamlit as st
import time
from explainer_chatbot_page import explainer_chatbot
from network_traffic_classifier_page import network_traffic_classifier


st.set_page_config(layout="wide")

# Initialize session states for tab tracking and other things
if "instruction" not in st.session_state :
    st.session_state.instruction=""
if "active_tab" not in st.session_state:
    st.session_state.active_tab='Network Traffic Classifier'
if 'container_shown' not in st.session_state :
    st.session_state.container_shown=False
if "messages_length" not in st.session_state :
    st.session_state.messages_length=0
if 'data_row' not in st.session_state :
    st.session_state.data_row=[]

if st.session_state.active_tab=='Network Traffic Classifier' :
    network_traffic_classifier()
elif st.session_state.active_tab=='Explainer & CyberSecurity Chatbot':
    explainer_chatbot()