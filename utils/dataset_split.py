import os
import yaml
import json
import argparse
import config
import random

import matplotlib.pyplot as plt
import matplotlib


# Get Parser from config
def get_parser():
    '''Parse the config file.'''
    print('ok')
    parser = argparse.ArgumentParser(description='PV Modules Failures Detection with GPT-4o.')
    parser.add_argument('--config', type=str,
                        default='config/config.yaml',
                        help='config file')
    parser.add_argument('opts',
                        default=None,
                        help='see config/config.yaml for all options',
                        nargs=argparse.REMAINDER)
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)

    return cfg


def split_data():
    '''
    Function to split the dataset in train and test for detection and classification
    '''

    args = get_parser()
    random.seed(args['seed'])
    json_path = args['dataset_json']

    # Opening JSON file
    f = open(json_path)

    # returns JSON object as a dictionary
    data = json.load(f)
    
    # Iterating through the json list
    no_anomaly = []
    anomaly = []
    anomaly_classes = dict()
    anomaly_classes['Cell'] = []
    anomaly_classes['Cell-Multi'] = []
    anomaly_classes['Cracking'] = []
    anomaly_classes['Hot-Spot'] = []
    anomaly_classes['Hot-Spot-Multi'] = []
    anomaly_classes['Shadowing'] = []
    anomaly_classes['Diode'] = []
    anomaly_classes['Diode-Multi'] = []
    anomaly_classes['Vegetation'] = []
    anomaly_classes['Soiling'] = []
    anomaly_classes['Offline-Module'] = []

    anomaly_classes_reduction = dict()
    anomaly_classes_reduction['Cell'] = []
    anomaly_classes_reduction['Cell-Multi'] = []
    anomaly_classes_reduction['Hot-Spot'] = []
    anomaly_classes_reduction['Hot-Spot-Multi'] = []
    anomaly_classes_reduction['Diode'] = []
    anomaly_classes_reduction['Diode-Multi'] = []
    anomaly_classes_reduction['Offline-Module'] = []

    anomaly_classes_reduction1 = dict()
    anomaly_classes_reduction1['Cell'] = []
    anomaly_classes_reduction1['Cell-Multi'] = []
    anomaly_classes_reduction1['Diode'] = []
    anomaly_classes_reduction1['Diode-Multi'] = []
    anomaly_classes_reduction1['Offline-Module'] = []

    data_anomaly = dict()
    data_anomaly_reduction = dict()
    data_anomaly_reduction1 = dict()
    for k in data.keys():
        if data[k]['anomaly_class'] == 'No-Anomaly':
            no_anomaly.append(k)
        else:
            anomaly.append(k)
            data_anomaly[k] = data[k]
            for c in anomaly_classes.keys():
                if data[k]['anomaly_class'] == c:
                    anomaly_classes[c].append(k)
           
            if data[k]['anomaly_class'] in anomaly_classes_reduction.keys():
                data_anomaly_reduction[k] = data[k]

            if data[k]['anomaly_class'] in anomaly_classes_reduction1.keys():
                data_anomaly_reduction1[k] = data[k]
                
            for c in anomaly_classes_reduction.keys():
                if data[k]['anomaly_class'] == c:
                    anomaly_classes_reduction[c].append(k)

            for c in anomaly_classes_reduction1.keys():
                if data[k]['anomaly_class'] == c:
                    anomaly_classes_reduction1[c].append(k)

    print('Reduction : 7 classes')
    for k in anomaly_classes_reduction.keys():
        print(k, len(anomaly_classes_reduction[k]))

    print('Reduction 1: 5 classes')
    for k in anomaly_classes_reduction1.keys():
        print(k, len(anomaly_classes_reduction1[k]))

    # Print and plot statistics 
    # plot_data(no_anomaly, anomaly, anomaly_classes, args['path_statistics'])

    # Divide the dataset in train and test
    train_detection, test_detection = detection_dataset(data, no_anomaly, anomaly)
    list_detection = {'train': list(train_detection.keys()), 'test': list(test_detection.keys())}
    
    train_classification, test_classification = classification_dataset(data_anomaly, anomaly_classes)
    list_classification =  {'train': list(train_classification.keys()), 'test': list(test_classification.keys())}
    
    train_classification_reduction, test_classification_reduction = classification_dataset_reduction(data_anomaly_reduction, anomaly_classes_reduction)
    list_classification_reduction =  {'train': list(train_classification_reduction.keys()), 'test': list(test_classification_reduction.keys())}
    
    train_classification_reduction1, test_classification_reduction1 = classification_dataset_reduction(data_anomaly_reduction1, anomaly_classes_reduction1)
    list_classification_reduction1 =  {'train': list(train_classification_reduction1.keys()), 'test': list(test_classification_reduction1.keys())}

    # Save the train and test set for detection and classification on the yaml file
    save_yaml(list_detection, 'binary-classification')
    save_yaml(list_classification, 'multi-classification')
    save_yaml(list_classification_reduction, 'multi-classification-reduction')
    save_yaml(list_classification_reduction1, 'multi-classification-reduction1')

    # Closing file
    f.close()


#
def detection_dataset(data, no_anomaly, anomaly):
    '''
    Function to divide in train and test the detection dataset.
    '''

    # Extract 80% of data for train set and the pther 20% for the test set
    train_detection = dict(random.sample(data.items(), int(len(data.keys())*0.8)))
    test_detection = {key: data[key] for key in data if key not in train_detection}

    # Count the occurencies for train and test for each class
    anomaly_train = 0
    no_anomaly_train = 0
    for k in train_detection.keys():
        if train_detection[k]['anomaly_class'] == 'No-Anomaly':
            no_anomaly_train += 1
        else:
            anomaly_train += 1

    anomaly_test = 0
    no_anomaly_test = 0
    for k in test_detection.keys():
        if test_detection[k]['anomaly_class'] == 'No-Anomaly':
            no_anomaly_test += 1
        else:
            anomaly_test += 1
    print('- STATISTICS DATASET DETECTION -')
    print('TRAIN - Anomaly : ', anomaly_train, ', No-Anomaly : ', no_anomaly_train)
    print('TEST - Anomaly : ', anomaly_test, ', No-Anomaly : ', no_anomaly_test)

    return train_detection, test_detection



def classification_dataset(data, anomaly_classes):
    '''
    Function to divide in train and test the classification dataset.
    '''

    # Extract 80% of data for train set and the other 20% for the test set
    train_classification = dict(random.sample(data.items(), int(len(data.keys())*0.8)))
    test_classification = {key: data[key] for key in data if key not in train_classification}

    # Count the occurencies for train and test for each class
    count_train = dict()
    for k in anomaly_classes.keys():
        count_train[k] = 0
        
    for k in train_classification.keys():
        for c in count_train.keys():
            if train_classification[k]['anomaly_class'] == c:
                count_train[c] += 1

    count_test = dict()
    for k in anomaly_classes.keys():
        count_test[k] = 0
        
    for k in test_classification.keys():
        for c in count_test.keys():
            if test_classification[k]['anomaly_class'] == c:
                count_test[c] += 1

    print('- STATISTICS DATASET CLASSIFICATION -')
    print('TRAIN - TEST')
    for c in anomaly_classes.keys():
        print(c, ':', count_train[c], '-', count_test[c])

    return train_classification, test_classification

def classification_dataset_reduction(data, anomaly_classes):
    '''
    Function to divide in train and test the classification dataset.
    '''

    # Extract 80% of data for train set and the other 20% for the test set
    print(data.keys(), type(data.keys()))
    train_classification = dict(random.sample(data.items(), int(len(data.keys())*0.8)))
    test_classification = {key: data[key] for key in data if key not in train_classification}

    # Count the occurencies for train and test for each class
    count_train = dict()
    for k in anomaly_classes.keys():
        count_train[k] = 0
        
    for k in train_classification.keys():
        for c in count_train.keys():
            if train_classification[k]['anomaly_class'] == c:
                count_train[c] += 1

    count_test = dict()
    for k in anomaly_classes.keys():
        count_test[k] = 0
        
    for k in test_classification.keys():
        for c in count_test.keys():
            if test_classification[k]['anomaly_class'] == c:
                count_test[c] += 1

    print('- STATISTICS DATASET CLASSIFICATION -')
    print('TRAIN - TEST')
    for c in anomaly_classes.keys():
        print(c, ':', count_train[c], '-', count_test[c])

    return train_classification, test_classification



def plot_data(no_anomaly, anomaly, anomaly_classes, path):
    '''
    Function to plot distribution of the occurences for the classes of each datasets train and test together.
    '''
    # Bar plot for anomaly and no_anomaly
    detection = plt.figure(figsize=(6, 10))
    plt.bar(['No-Anomaly', 'Anomaly'], [len(no_anomaly), len(anomaly)], width=0.2)
    plt.title('No-Anomaly vs Anomaly')
    plt.xlabel('Category')
    plt.ylabel('Occurrences')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{path}/statistics_detection.pdf", dpi=50, bbox_inches="tight")
    plt.close()

    # Bar plot for anomaly_classes
    plt.figure(figsize=(10, 10))
    classes = list(anomaly_classes.keys())
    occurrences = [len(anomaly_classes[c]) for c in classes]
    plt.bar(classes, occurrences)
    plt.title('Occurrences of Anomaly Classes')
    plt.xlabel('Classes')
    plt.ylabel('Occurrences')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
    plt.savefig(f"{path}/statistics_classification.pdf", dpi=50, bbox_inches="tight")
    plt.close()



def save_yaml(list, task):
    '''
    Function to save the .yaml file with the id images for the training and test sets for detection and classification tasks. 
    The files are structured in this way:
    test:
        - 1
        - 2
    train:
        - 3
        - 4
    '''
    

    # Function to save the yaml file as:
    if task is None:
        raise Exception('ERROR : Specify the type of task between detection and classification')
    elif task == 'binary-classification':
        path = 'config/dataset_binary.yaml'
    elif task == 'multi-classification':
        path = 'config/dataset_multi.yaml'
    elif task == 'multi-classification-reduction':
        path = 'config/dataset_multi_reduction.yaml'
    elif task == 'multi-classification-reduction1':
        path = 'config/dataset_multi_reduction1.yaml'
    else:
        raise Exception('Error in task variable')
    
    with open(path, 'w') as file:
        yaml.dump(list, file)
    
    return

# To run matplotlib in a no GUI environment (comment if needed) 
matplotlib.use('Agg')

# Run the code
split_data()


