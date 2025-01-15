import numpy as np
import shap
import json
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from unsloth import FastLanguageModel
from langchain_core.messages import HumanMessage

def get_key(dict,value) :
    return [key for key in dict.keys() if dict[key]==value][0]
def sort_get_best(shap_values,features) :
    feat_dict={features[i]:shap_values[i] for i in range(0,len(features))}
    sorted_shap = np.argsort(np.abs(shap_values))
    result_shap = shap_values[sorted_shap][::-1][:10]
    feature_importances={get_key(feat_dict,value):value for value in result_shap if value>=0}
    return feature_importances
def make_prompt(data_row,classifier,metadata,best_shap_values) :
    classes={0:'Normal Traffic',1 : 'DoS attack',2: 'DDoS attack',3: 'OS Fingerprint attack',4: 'Service Scan attack',5 : 'Keylogging attack',6 : 'Data Exfiltration attack'}
    #making the prompts
    flags_state=['flgs_e', 'flgs_e s', 'flgs_e dS', 'flgs_e g', 'flgs_e *', 'flgs_eU',
        'flgs_e &', 'flgs_e    F', 'flgs_e r']
    protos=['proto_tcp', 'proto_udp',
        'proto_icmp', 'proto_arp', 'proto_ipv6-icmp', 'proto_rarp',
        'proto_igmp']
    s_addr=['saddr_192.168.100.149', 'saddr_192.168.100.148',
        'saddr_192.168.100.150', 'saddr_192.168.100.147', 'saddr_192.168.100.3',
        'saddr_192.168.100.7', 'saddr_192.168.100.5', 'saddr_192.168.100.4',
        'saddr_192.168.100.1']
    d_addr=['daddr_192.168.100.149', 'daddr_192.168.100.150',
        'daddr_192.168.100.147', 'daddr_192.168.100.3', 'daddr_192.168.100.7',
        'daddr_192.168.100.6', 'daddr_192.168.100.5', 'daddr_192.168.100.4']
    d_type=['daddr_private','daddr_external']
    states=['state_RST','state_CON', 'state_NRS', 'state_ACC', 'state_MAS']
    transaction_info=['pkts', 'bytes','dur', 'mean',
        'min', 'dpkts', 'dbytes', 'tnp_per_dport', 'ar_p_proto_p_dstip',
        'ar_p_proto_p_sport', 'pkts_p_state_p_protocol_p_destip']

    #network traffic and classifier decision
    prompt="Generate a concise yet profound explanation of a network classifier's decision regarding whether the analyzed network traffic is normal or represents a type of attack. Use the provided SHAP values to highlight the classifier's reasoning while also considering key traffic details such as transaction state, source and destination addresses, and transaction-specific information. Ensure the explanation is insightful and well-rounded.\ninput : {\n"
    prompt=prompt+"Network Traffic :\n\n"
    columns=list(data_row.index)
    columns.append('additional_info')
    columns.append('attack_type')
    y_pred=classifier.predict([data_row])[0]
    for feature in columns :
        if feature=="additional_info" :
            prompt=prompt+f"\nConsider the following information as well: \n-The IP addresses 192.168.100.147, 192.168.100.148, 192.168.100.149, and 192.168.100.150 are identified as botnets, meaning that attacks are likely to originate from these IPs. \n-The IP address 192.168.100.3 corresponds to the server within the network which is usually targeted by network attacks.\n"
        if feature=="attack_type" :
            prompt=prompt+f"\nClassifier Decision : {classes[y_pred]}\n"
        elif feature in flags_state :
            if data_row[feature]==1.0 :
                prompt=prompt+f"Flow state flags: {metadata[feature]}\n"
        elif feature in protos :
            if data_row[feature]==1.0 :
                prompt=prompt+f"Transaction protocol : {metadata[feature]}\n"
        elif feature in s_addr :
            if data_row[feature]==1.0 :
                prompt=prompt+f"Source IP address : {metadata[feature]}\n"
        elif feature in d_addr :
            if data_row[feature]==1.0 :
                prompt=prompt+f"Destination IP address : {metadata[feature]}\n"
        elif feature in d_type :
            if data_row[feature]==1.0 :
                prompt=prompt+f"Destination address type : {metadata[feature]}\n"
        elif feature in states :
            if data_row[feature]==1.0 :
                prompt=prompt+f"-Transaction State : {metadata[feature]}\n"
        elif feature in transaction_info :
            if "\nTransaction information :\n" in prompt :
                prompt=prompt+f"-{metadata[feature]} : {round(data_row[feature],2)}\n"
            else :
                prompt=prompt+"\nTransaction information :\n"
                prompt=prompt+f"-{metadata[feature]} : {round(data_row[feature],2)}\n"
    #shap values
    prompt=prompt+"\nSHAP Values that represent how much a particular feature influenced the final decision:\n"
    for feature in best_shap_values.keys() :
        if feature in flags_state :
            if data_row[feature]==1.0 :
                prompt=prompt+ f"-SHAP Value for Flow state flags : {metadata[feature]} : {round(best_shap_values[feature],2)} (Feature value: {data_row[feature]})\n"
        elif feature in protos :
            if data_row[feature]==1.0 :
                prompt=prompt+ f"-SHAP Value for Transaction protocol : {metadata[feature]} : {round(best_shap_values[feature],2)}. (Feature value: {data_row[feature]})\n"
        elif feature in s_addr :
            if data_row[feature]==1.0 :
                prompt=prompt+ f"-SHAP Value for Source IP Address ({metadata[feature]}) : {round(best_shap_values[feature],2)}. (Feature value: {data_row[feature]})\n"
        elif feature in d_addr :
            if data_row[feature]==1.0 :
                prompt=prompt+ f"-SHAP Value for Destination IP Address ({metadata[feature]}) : {round(best_shap_values[feature],2)}. (Feature value: {data_row[feature]})\n"
        elif feature in d_type :
            prompt=prompt+ f"-SHAP Value for Destination Address Type ({metadata[feature]}) : {round(best_shap_values[feature],2)}. (Feature value: {data_row[feature]})\n"
        elif feature in states :
            if data_row[feature]==1.0 :
                prompt=prompt+ f"-SHAP Value for Transaction state ({metadata[feature]}) : {round(best_shap_values[feature],2)}. (Feature value: {data_row[feature]})\n"
        else :
            prompt=prompt+ f"-SHAP Value for {metadata[feature]} : {round(best_shap_values[feature],2)}. (Feature value: {data_row[feature]})\n"
    prompt=prompt[:-1]+'}'
    return prompt

def llm_data_pipeline(data_row,classifier) :
    """
    This function returns the prompt that will be passed to the ExplainerLLM via the data_row fed to the classifier.
    """
    with open("./files/prompt_features.json",'r') as f :
        metadata=json.load(f)
    #classifier=joblib.load("/teamspace/studios/this_studio/LLM-for-IDS-log-analysis/app/classifier.pkl")
    explainer=shap.TreeExplainer(classifier)
    shap_values=explainer.shap_values([data_row])[0,:,classifier.predict([data_row])[0]]
    best_shap_values=sort_get_best(shap_values,data_row.index)
    llm_prompt=make_prompt(data_row,classifier,metadata,best_shap_values)
    prompt=llm_prompt.split("input")[1].replace("}","")[5:-1]
    return prompt

def load_model() :
    """
    This function loads the fine-tuned Cybersecurity-Llama-3.2-3B-Instruct alongside its tokenizer.
    """
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "./fine-tuned-llm-llama3-instruct-cyberexplain-final", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    # Create a text-generation pipeline using the PEFT model
    model_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1000000,
    )
    cyberexplainer = HuggingFacePipeline(pipeline=model_pipeline)


    return cyberexplainer
def load_model_shap() :
    """
    This function loads the fine-tuned Cybersecurity-Llama-3.2-3B-Instruct alongside its tokenizer.
    """
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "./fine-tuned-llm-llama3-instruct-cyberexplain-final", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    # Create a text-generation pipeline using the PEFT model
    model_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1300,
    )
    cyberexplainer = HuggingFacePipeline(pipeline=model_pipeline)


    return cyberexplainer
def format_prompt(llm_prompt,convo=False) :
    if not convo :
        return """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nGenerate a concise yet profound explanation of a network classifier's decision regarding whether the analyzed network traffic is normal or represents a type of attack. Use the provided SHAP values to highlight the classifier's reasoning while also considering key traffic details such as transaction state, source and destination addresses, and transaction-specific information. Ensure the explanation is insightful and well-rounded.\n\n### Input:\n{}\n\n### Response:\n""".format(llm_prompt) 
    else :
        return """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nYou are an assistant named CyberExplainer, a helpful and creative cybersecurity expert made by a data scientist named Anass EL MANSOURI. Assist the user by providing expertise in Cybersecurity. Don't generate user questions and only answer the question given to you.\n\n### Input:\n{}\n\n### Response:\n""".format(llm_prompt)    
def get_response(llm_prompt,chain,conv=False) :
    """
    This function loads the fine-tuned LLM, sets up the prediction to run on GPU and then returns the generated response.
    """
    config = {"configurable": {"session_id": "abc2"}}
    response = chain.invoke(
                    [HumanMessage(content=format_prompt(llm_prompt,conv))],
                    config=config,
                )
    # Run the chain with a sample input
    #response = chain.run(llm_prompt)
    return response

def clean_response(response : str, conversation : bool) :
    """
    Function for cleaning the repsonses generated by the ExplainerLLM
    """
    response=response.split("### Response:")[-1].strip()
    return response

def get_final_response(llm_prompt,chain,conversation=False,conv=False):
    """
    This function cleans the response if it's not completed.
    """
    unclean_response=get_response(llm_prompt,chain,conv)
    return clean_response(unclean_response,conversation)

def get_final_response_nn(llm_prompt,chain,conversation=False):
    """
    This function cleans the response if it's not completed.
    """
    unclean_response=get_response(llm_prompt,chain)
    response=clean_response(unclean_response,conversation)
    if not response.endswith(('.', '!', '?')):
        last_period = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
        if last_period != -1:
            response = response[:last_period + 1]
    return response