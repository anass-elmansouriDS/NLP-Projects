import pandas as pd

def make_classifier_data(variables,classifier_features,abbv) :
    """
    Constructs a data row in the right format for the classifier from the variables fetched from the user in the UI.

    Args :
       -variables (list) : Feature values fetched from the user's input
       -classifier_features (list) : Names of the features that the CyberClassifier accepts, must be formated in the same way that was introduced in training.
       -abbv (dict) : List of abbreviations so that we can map the variable names to the classifier features. 
    """
    data_row={}
    for column in classifier_features :
        data_row[column]=[0.0]
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

def make_prediction(data_row,classifier) :
    """
    Returns the classifier's prediction on the data row input by the user.

    Args : 
        -data_row : the formated data row returned by the function make_classifier_data.
        -classifier : the loaded CyberClassifier model.
    """
    return classifier.predict(data_row)[0]
