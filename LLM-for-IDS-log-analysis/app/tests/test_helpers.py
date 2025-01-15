from helpers import *
from random import choice
import joblib
def test_helpers() :
    src_addrs=choice(["192.168.100.149",'192.168.100.148','192.168.100.150','192.168.100.147','192.168.100.1' ,'192.168.100.3' ,'192.168.100.4' ,'192.168.100.7' ,'192.168.100.5'])
    flgs = choice(["ESTABLISHED", "ESTABLISHED | SYN_SENT", "ESTABLISHED | SYN_RECEIVED", "ESTABLISHED | FIN_WAIT_1","ESTABLISHED | ACK","ESTABLISHED | URGENT","ESTABLISHED | PUSH","ESTABLISHED | FIN_WAIT_2","ESTABLISHED | RESET"])
    pkts=2.0
    bytes=120.0
    dur=0.004128
    mean=0.004128
    dbytes=60.000000
    tnp_per_dport=8.000000
    ar_p_proto_p_dstip=939.965691
    dest_addrs=choice(["192.168.100.149",'192.168.100.150','192.168.100.147','192.168.100.6','192.168.100.3' ,'192.168.100.4' ,'192.168.100.7' ,'192.168.100.5'])
    proto = choice(["TCP", "UDP", "ICMP", "IPV6-ICMP","ARP","RARP","IGMP"])
    states=choice(["RESET",'CONNECTED','NO_RESPONSE','ACCEPTED','MASQUERADE'])
    dst_types=choice(['Private','External'])
    min_dur=0.004128
    dpkts=1.0
    ar_p_proto_p_sport=564.567137
    pkts_p_state_p_protocol_p_destip=6482.0
    variables={'saddr' :src_addrs,'daddr':dest_addrs,'proto':proto,'flgs':flgs,'state':states,'daddr_type':dst_types,'min' :min_dur,'pkts' :pkts,'dpkts' :dpkts,'bytes':bytes,'dbytes':dbytes,'dur':dur, 'mean':mean, 'tnp_per_dport':tnp_per_dport, 'ar_p_proto_p_dstip':ar_p_proto_p_dstip,'ar_p_proto_p_sport':ar_p_proto_p_sport, 'pkts_p_state_p_protocol_p_destip':pkts_p_state_p_protocol_p_destip}
    row=make_classifier_data(variables)
    assert str(type(row))=="<class 'pandas.core.series.Series'>"
    assert len(row.index)==51
    classifier=joblib.load("/teamspace/studios/this_studio/NLP-Projects/LLM-for-IDS-log-analysis/app/files/classifier.pkl")
    pred=make_pred(classifier,row)
    assert str(type(pred))=="<class 'int'>"
