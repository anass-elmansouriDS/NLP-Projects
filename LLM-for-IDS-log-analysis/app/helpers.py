import pandas as pd
def make_classifier_data(variables) :
    classifier_features=['flgs_e', 'flgs_e s', 'flgs_e dS', 'flgs_e g', 'flgs_e *', 'flgs_eU',
       'flgs_e &', 'flgs_e    F', 'flgs_e r', 'proto_tcp', 'proto_udp',
       'proto_icmp', 'proto_arp', 'proto_ipv6-icmp', 'proto_rarp',
       'proto_igmp', 'saddr_192.168.100.149', 'saddr_192.168.100.148',
       'saddr_192.168.100.150', 'saddr_192.168.100.147', 'saddr_192.168.100.3',
       'saddr_192.168.100.7', 'saddr_192.168.100.5', 'saddr_192.168.100.4',
       'saddr_192.168.100.1', 'daddr_192.168.100.149', 'daddr_192.168.100.150',
       'daddr_192.168.100.147', 'daddr_192.168.100.3', 'daddr_192.168.100.7',
       'daddr_192.168.100.6', 'daddr_192.168.100.5', 'daddr_192.168.100.4',
       'daddr_private', 'daddr_external', 'pkts', 'bytes', 'state_RST',
       'state_CON', 'state_NRS', 'state_ACC', 'state_MAS', 'dur', 'mean',
       'min', 'dpkts', 'dbytes', 'tnp_per_dport', 'ar_p_proto_p_dstip',
       'ar_p_proto_p_sport', 'pkts_p_state_p_protocol_p_destip']
    data_row={}
    for column in classifier_features :
        data_row[column]=[0.0]
    abbv={'flgs' : {"ESTABLISHED" : "e",'ESTABLISHED | SYN' : 'e s','ESTABLISHED | SYN_RECEIVED' : 'e dS','ESTABLISHED | FIN_WAIT_2' : 'e    F','ESTABLISHED | FIN_WAIT_1' : 'e g','ESTABLISHED | ACK' : 'e *','ESTABLISHED | URGENT' :'eU','ESTABLISHED | PUSH' :'e &','ESTABLISHED | RESET' :'e r'},
    'state' : {'RESET' : 'RST','CONNECTED':'CON','NO_RESPONSE':'NRS','ACCEPTED':'ACC','MASQUERADE':'MAS'}
    }
    for column in classifier_features :
        if 'flgs' in column :
            data_row[f"flgs_{abbv['flgs'][variables['flgs']]}"]=[1.0]
        elif 'saddr' in column :
            data_row[f"saddr_{variables['saddr']}"]=[1.0]
        elif 'daddr' in column and column not in ['daddr_private','daddr_external']:
            data_row[f"daddr_{variables['daddr']}"]=[1.0]
        elif column in ['daddr_private','daddr_external'] :
            data_row[f"daddr_{variables['daddr_type'].lower()}"]=[1.0]
        elif 'proto' in column :
            data_row[f"proto_{variables['proto'].lower()}"]=[1.0]
        elif 'state' in column :
            data_row[f"state_{abbv['state'][variables['state']]}"]=[1.0]
        else :
            data_row[column]=[variables[column]]
    return pd.DataFrame(data_row).iloc[0]
def make_pred(data_row,classifier) :
    return classifier.predict(data_row)[0]
