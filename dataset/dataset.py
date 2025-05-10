import os
import torch
import yaml
import json

import numpy as np

from PIL import Image
from config.dataset_learning_map import *

class ImageLoader_ResNet(torch.utils.data.Dataset):
    '''Dataloader for images.'''

    def __init__(self, config_file='config/config.yaml', 
                 split='train', task='detection', aug=False,
                 input_color=False, transform=None
                 ):
        super().__init__()

        """Load data from given dataset directory."""
        print(config_file)
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        json_path = self.config['dataset']['dataset_json']

        # Opening JSON file
        j = open(json_path)

        # returns JSON object as a dictionary
        self.data = json.load(j)
        j.close()

        self.dataset_path = self.config['dataset']['dataset_path']
        self.split = self.config['dataset']['split']
        self.aug = self.config['dataset']['aug']
        self.color = self.config['dataset']['color']
        self.task = self.config['general']['task']
        self.transform = transform

        if not os.path.exists(self.config['dataset']["dataset_path"]):
            raise RuntimeError("Images directory missing: " + self.config['dataset']["dataset_path"])
       
        if self.task is None:
            raise Exception('Specify the task between binary-classification, multi-classification and multi-classification-reduction in the config file')
        else:

            # open the yaml file for dataset based on the task
            if self.task == 'binary-classification':
                self.yaml_file = 'config/dataset_binary.yaml'
            elif self.task == 'multi-classification':
                self.yaml_file = 'config/dataset_multi.yaml'
            elif self.task == 'multi-classification-reduction':
                self.yaml_file = 'config/dataset_multi_reduction.yaml'
            elif self.task == 'multi-classification-reduction1':
                self.yaml_file = 'config/dataset_multi_reduction1.yaml'
            else:
                AttributeError(self.task)

            with open(self.yaml_file, "r") as y:
                self.config_dataset = yaml.safe_load(y)

            self.learning_map_path = self.config['dataset']['dataset_learning_map']
            with open(self.learning_map_path, "r") as lm:
                self.learning_map_file = yaml.safe_load(lm)
        
            if self.learning_map_file is None:
                raise Exception('Learning Map file is None')
            else:
                if self.task == 'binary-classification':
                    # learning map for detection
                    # learning_map = binary_learning_map
                    # self.learning_map_function = np.vectorize(lambda x: learning_map[x])
                    self.learning_map_function = binary_learning_map
                    # inv_learning_map = binary_inv_learning_map
                    # self.inv_learning_map_function = np.vectorize(lambda x: inv_learning_map[x])
                    self.inv_learning_map_function = binary_inv_learning_map
                elif self.task == 'multi-classification':
                    # learning map for detection
                    self.learning_map_function = multi_learning_map
                    # self.learning_map_function = np.vectorize(lambda x: learning_map[x])

                    self.inv_learning_map_function = multi_inv_learning_map
                    # self.inv_learning_map_function = np.vectorize(lambda x: inv_learning_map[x])
                elif self.task == 'multi-classification-reduction':
                    # learning map for detection
                    self.learning_map_function = multi_reduction_learning_map
                    # self.learning_map_function = np.vectorize(lambda x: learning_map[x])

                    self.inv_learning_map_function = multi_reduction_inv_learning_map
                    # self.inv_learning_map_function = np.vectorize(lambda x: inv_learning_map[x])
                elif self.task == 'multi-classification-reduction1':
                    # learning map for detection
                    self.learning_map_function = multi_reduction_learning_map1
                    # self.learning_map_function = np.vectorize(lambda x: learning_map[x])

                    self.inv_learning_map_function = multi_reduction_inv_learning_map1
                    # self.inv_learning_map_function = np.vectorize(lambda x: inv_learning_map[x])
                else:
                    AttributeError(self.task)

            # if self.learning_map_file is None:
            #     raise Exception('Learning Map file is None')
            # else:
            #     if self.task == 'binary-classification':
            #         # learning map for detection
            #         learning_map = self.learning_map_file['binary_learning_map']
            #         self.learning_map_function = np.vectorize(lambda x: learning_map[x])

            #         inv_learning_map = self.learning_map_file['binary_inv_learning_map']
            #         self.inv_learning_map_function = np.vectorize(lambda x: inv_learning_map[x])
            #     elif self.task == 'multi-classification':
            #         # learning map for detection
            #         learning_map = self.learning_map_file['multi_learning_map']
            #         self.learning_map_function = np.vectorize(lambda x: learning_map[x])

            #         inv_learning_map = self.learning_map_file['multi_inv_learning_map']
            #         self.inv_learning_map_function = np.vectorize(lambda x: inv_learning_map[x])
            #     else:
            #         AttributeError(self.task)

        self.files = []
        for ids in self.config_dataset[self.split]:
            im_path = self.data[ids]['image_filepath']
            anomaly_class = self.data[ids]['anomaly_class']
            self.files.append({'id': ids, 'path': f'{self.dataset_path}/{im_path}', 'label': self.learning_map_function[f'{anomaly_class}']})


    def __getitem__(self, index):
        i = self.files[index]
        im_id = torch.Tensor(int(i['id']))
        im_path = i['path']
        im_label = torch.tensor(i['label'], dtype=torch.long)
        index = torch.Tensor(index)

        if not self.color:
            image = Image.open(im_path).convert('RGB')
        elif self.color:
            image = Image.open(im_path).convert('RGB')
        else:
            AttributeError(self.color)

        if self.transform:
            image = self.transform(image)
        
        # return im_id, im_path, im_label, index, image
        return im_label, image
    
    def __len__(self):
        return len(self.files) 