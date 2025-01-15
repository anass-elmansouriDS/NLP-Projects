import streamlit as st
import time
from helpers import make_classifier_data,make_pred
from llm_helpers import llm_data_pipeline,get_final_response,load_model,load_model_shap
import joblib
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
import torch
st.set_page_config(layout="wide")
# Create a single-line text input
# Create tabs
# Initialize session state for tab tracking
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

# Define a function to change the active tab
def switch_tab(tab_name):
    return tab_name
if st.session_state.active_tab=='Network Traffic Classifier' :
    st.subheader("Network Traffic Classifier")
    col1,col2=st.columns(2)
    with col1 :
        src_addrs=["192.168.100.149",'192.168.100.148','192.168.100.150','192.168.100.147','192.168.100.1' ,'192.168.100.3' ,'192.168.100.4' ,'192.168.100.7' ,'192.168.100.5']
        source_addr = st.selectbox("Source Address :",src_addrs)
        flgs = ["ESTABLISHED", "ESTABLISHED | SYN_SENT", "ESTABLISHED | SYN_RECEIVED", "ESTABLISHED | FIN_WAIT_1","ESTABLISHED | ACK","ESTABLISHED | URGENT","ESTABLISHED | PUSH","ESTABLISHED | FIN_WAIT_2","ESTABLISHED | RESET"]
        flgs_in = st.selectbox("Flow state flags seen in transactions :", flgs)
        pkts=st.number_input('Total count of packets in transaction :',value=2.0)
        bytes=st.number_input('Total number of bytes in transaction :',value=120.0)
        dur=st.number_input('Record total duration :',value=0.004128)
        mean=st.number_input('Average duration of aggregated records :',value=0.004128)
        dbytes=st.number_input('Destination-to-source byte count :',value=60.000000)
        tnp_per_dport=st.number_input('Total Number of packets per destination port :',value=8.000000)
        ar_p_proto_p_dstip=st.number_input('Average rate per protocol per Destination IP :',value=939.965691)

    with col2 :
        dest_addrs=["192.168.100.149",'192.168.100.150','192.168.100.147','192.168.100.6','192.168.100.3' ,'192.168.100.4' ,'192.168.100.7' ,'192.168.100.5']
        dest_addr = st.selectbox("Destination Address :",dest_addrs)
        proto = ["TCP", "UDP", "ICMP", "IPV6-ICMP","ARP","RARP","IGMP"]
        proto_in = st.selectbox("Transaction protocol present in network flow :", proto)
        states=["RESET",'CONNECTED','NO_RESPONSE','ACCEPTED','MASQUERADE']
        state=st.selectbox("Transaction state :",states)
        dst_types=['Private','External']
        dst_type=st.selectbox("Destination Address Type :",dst_types)
        min_dur=st.number_input('Minimum duration of aggregated records :',value=0.004128)
        dpkts=st.number_input("Destination-to-source packet count :",value=1.0)
        ar_p_proto_p_sport=st.number_input('Average rate per protocol per sport :',value=564.567137)
        pkts_p_state_p_protocol_p_destip=st.number_input('Number of packets grouped by state of flows and protocols per destination IP :',value=6482.0)
        classes={0:'Normal Traffic',1 : 'DoS attack',2: 'DDoS attack',3: 'OS Fingerprint attack',4: 'Service Scan attack',5 : 'Keylogging attack',6 : 'Data Exfiltration attack'}
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
    if 'cyberexplainer_shap' not in st.session_state  :
        st.session_state.cyberexplainer_shap=load_model_shap()
    if 'classifier' not in st.session_state :
        st.session_state.classifier=joblib.load("./files/classifier.pkl")
    if 'chain_shap' not in st.session_state :
        store = {}
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = InMemoryChatMessageHistory()
            return store[session_id]
        st.session_state.chain_shap = RunnableWithMessageHistory(st.session_state.cyberexplainer_shap, get_session_history)
    terminal_container.markdown("<div class='terminal'>The network traffic classifier and the explainer LLM are ready.</div>", unsafe_allow_html=True)
    st.write(" ")
    #st.write(st.session_state.response)
    if st.button('Classify!') :
        variables={'saddr' :source_addr,'daddr':dest_addr,'proto':proto_in,'flgs':flgs_in,'state':state,'daddr_type':dst_type,'min' :min_dur,'pkts' :pkts,'dpkts' :dpkts,'bytes':bytes,'dbytes':dbytes,'dur':dur, 'mean':mean, 'tnp_per_dport':tnp_per_dport, 'ar_p_proto_p_dstip':ar_p_proto_p_dstip,
       'ar_p_proto_p_sport':ar_p_proto_p_sport, 'pkts_p_state_p_protocol_p_destip':pkts_p_state_p_protocol_p_destip}
        data_row=make_classifier_data(variables)
        decision=make_pred([data_row],st.session_state.classifier)
        st.session_state.data_row=data_row
        llm_prompt=llm_data_pipeline(data_row,st.session_state.classifier)
        output=""
        for char in "Processing the inputs..." :
            output+=char 
            terminal_container.markdown(f"<div class='terminal'>{output}</div>", unsafe_allow_html=True)
            time.sleep(0.08)
        time.sleep(3)
        output=""
        for char in "Calculating the results..." :
            output+=char 
            terminal_container.markdown(f"<div class='terminal'>{output}</div>", unsafe_allow_html=True)
            time.sleep(0.08)
        time.sleep(2)
        output=""
        for char in "Making the prediction..." :
            output+=char 
            terminal_container.markdown(f"<div class='terminal'>{output}</div>", unsafe_allow_html=True)
            time.sleep(0.08)
        output=""
        for char in "Generating the explanation..." :
            output+=char
            terminal_container.markdown(f"<div class='terminal'>{output}</div>", unsafe_allow_html=True)
            time.sleep(0.08)
        if "response" not in st.session_state :
            st.session_state.response=get_final_response(llm_prompt,st.session_state.chain_shap)
        if "attack" not in st.session_state :
            st.session_state.attack=classes[decision]
        decision_str=f"The network traffic classifier classified this traffic as a {classes[decision]}."
        output=""
        for char in decision_str :
            output+=char 
            terminal_container.markdown(f"<div class='terminal'>{output}</div>", unsafe_allow_html=True)
            time.sleep(0.05)  
    #st.write(st.session_state.response.replace("\n\n",""))  
    if st.button("Learn more about the classifier's decision"):
        st.session_state.active_tab=switch_tab("Explainer & CyberSecurity Chatbot")
        st.rerun() 
elif st.session_state.active_tab=='Explainer & CyberSecurity Chatbot':
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "user", "text": "Explain the classifier's decision"},{"role": "assistant", "text": st.session_state.response}]
    st.subheader("Cybersecurity chatbot")
    col5,col6,col7,col8,col9=st.columns(5)
    with col5 :
        if st.button("New Chat") :
            st.session_state.messages=[] 
    with col9 :
        if st.button('Back to the network classifier'):
            st.session_state.active_tab=switch_tab("Network Traffic Classifier")
            st.rerun()
    # Function to simulate chatbot response
    def get_chatbot_response(user_message):
        # You can customize this to have a more intelligent response
        return get_final_response(user_message,st.session_state.chain,True,True)
    
    #show messages history
    for message in st.session_state.messages :
        with st.chat_message(message["role"]):
            st.markdown(message["text"])            
    # User input for chatbot
    if prompt := st.chat_input("Talk to cyberexplainer :") :
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        if 'cyberexplainer' not in st.session_state :
            del st.session_state.cyberexplainer_shap
            torch.cuda.empty_cache()
            st.session_state.cyberexplainer=load_model()
        if 'chain' not in st.session_state :
            store = {}
            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in store:
                    store[session_id] = InMemoryChatMessageHistory()
                return store[session_id]
            st.session_state.chain = RunnableWithMessageHistory(st.session_state.cyberexplainer, get_session_history)
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
