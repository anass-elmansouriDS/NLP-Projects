import streamlit as st
import time
import json
import joblib
from helpers import make_classifier_data,make_prediction
from llm_helpers import llm_data_pipeline,get_final_response,load_model
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory


def network_traffic_classifier() :
    """
    Sets up the network traffic classification streamlit page. 
    """
    st.subheader("Network Traffic Classifier")

    #load the constants
    with open("/teamspace/studios/this_studio/NLP-Projects/LLM-for-IDS-log-analysis/app/files/constants.json",'r') as f :
        constants=json.load(f)
    
    col1,col2=st.columns(2)
    
    with col1 :
        source_addr = st.selectbox("Source Address :", constants["Traffic_classifier_constants"]["src_addrs"])
        flgs_in = st.selectbox("Flow state flags seen in transactions :", constants["Traffic_classifier_constants"]["flgs"])
        pkts=st.number_input('Total count of packets in transaction :',value=2.0)
        bytes=st.number_input('Total number of bytes in transaction :',value=120.0)
        dur=st.number_input('Record total duration :',value=0.004128)
        mean=st.number_input('Average duration of aggregated records :',value=0.004128)
        dbytes=st.number_input('Destination-to-source byte count :',value=60.000000)
        tnp_per_dport=st.number_input('Total Number of packets per destination port :',value=8.000000)
        ar_p_proto_p_dstip=st.number_input('Average rate per protocol per Destination IP :',value=939.965691)

    with col2 :
        dest_addr = st.selectbox("Destination Address :",constants["Traffic_classifier_constants"]["dest_addrs"])
        proto_in = st.selectbox("Transaction protocol present in network flow :", constants["Traffic_classifier_constants"]["proto"])
        state=st.selectbox("Transaction state :",constants["Traffic_classifier_constants"]["states"])
        dst_type=st.selectbox("Destination Address Type :",constants["Traffic_classifier_constants"]["dst_types"])
        min_dur=st.number_input('Minimum duration of aggregated records :',value=0.004128)
        dpkts=st.number_input("Destination-to-source packet count :",value=1.0)
        ar_p_proto_p_sport=st.number_input('Average rate per protocol per sport :',value=564.567137)
        pkts_p_state_p_protocol_p_destip=st.number_input('Number of packets grouped by state of flows and protocols per destination IP :',value=6482.0)
    # CSS for terminal style
    st.markdown("""
        <style>
        .terminal {
            background-color: black;
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            height: 80px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create a container for the terminal output
    terminal_container = st.empty()
    terminal_container.markdown("<div class='terminal'>Loading the network traffic classifier and the explainer LLM...</div>", unsafe_allow_html=True)
    if 'cyberexplainer' not in st.session_state  :
        try :
            st.session_state.cyberexplainer=load_model()
        except :
            st.rerun()
    if 'classifier' not in st.session_state :
        st.session_state.classifier=joblib.load("/teamspace/studios/this_studio/NLP-Projects/LLM-for-IDS-log-analysis/app/files/classifier.pkl")
    if 'chain' not in st.session_state :
        store = {}
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = InMemoryChatMessageHistory()
            return store[session_id]
        st.session_state.chain = RunnableWithMessageHistory(st.session_state.cyberexplainer, get_session_history)
    terminal_container.markdown("<div class='terminal'>The network traffic classifier and the explainer LLM are ready.</div>", unsafe_allow_html=True)
    st.write(" ")
    #st.write(st.session_state.response)
    if st.button('Classify!') :
        variables={'saddr' :source_addr,'daddr':dest_addr,'proto':proto_in,'flgs':flgs_in,'state':state,'daddr_type':dst_type,'min' :min_dur,'pkts' :pkts,'dpkts' :dpkts,'bytes':bytes,'dbytes':dbytes,'dur':dur, 'mean':mean, 'tnp_per_dport':tnp_per_dport, 'ar_p_proto_p_dstip':ar_p_proto_p_dstip,
        'ar_p_proto_p_sport':ar_p_proto_p_sport, 'pkts_p_state_p_protocol_p_destip':pkts_p_state_p_protocol_p_destip}
        
        output=""
        for char in "Processing the inputs..." :
            output+=char 
            terminal_container.markdown(f"<div class='terminal'>{output}</div>", unsafe_allow_html=True)
            data_row=make_classifier_data(variables,constants["Traffic_classifier_constants"]["classifier_features"],constants["Traffic_classifier_constants"]["abbv"])
            st.session_state.data_row=data_row
            time.sleep(0.08)
        time.sleep(3)
        output=""
        for char in "Calculating the results..." :
            output+=char 
            terminal_container.markdown(f"<div class='terminal'>{output}</div>", unsafe_allow_html=True)
            llm_prompt=llm_data_pipeline(data_row,st.session_state.classifier,constants)
            time.sleep(0.08)
        time.sleep(2)
        output=""
        for char in "Making the prediction..." :
            output+=char 
            terminal_container.markdown(f"<div class='terminal'>{output}</div>", unsafe_allow_html=True)
            decision=make_prediction([data_row],st.session_state.classifier)
            time.sleep(0.08)
        output=""
        for char in "Generating the explanation..." :
            output+=char
            terminal_container.markdown(f"<div class='terminal'>{output}</div>", unsafe_allow_html=True)
            time.sleep(0.08)
        if "response" not in st.session_state :
            st.session_state.response=get_final_response(llm_prompt,st.session_state.chain,conversation=False)
        if "attack" not in st.session_state :
            st.session_state.attack=constants["Traffic_classifier_constants"]["classes"][str(decision)]
        decision_str=f"The network traffic classifier classified this traffic as a {st.session_state.attack}."
        output=""
        for char in decision_str :
            output+=char 
            terminal_container.markdown(f"<div class='terminal'>{output}</div>", unsafe_allow_html=True)
            time.sleep(0.05)  
    #st.write(st.session_state.response.replace("\n\n",""))  
    if st.button("Learn more about the classifier's decision"):
        st.session_state.active_tab="Explainer & CyberSecurity Chatbot"
        st.rerun()