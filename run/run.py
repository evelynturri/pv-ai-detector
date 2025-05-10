import os
import copy
import torch
import argparse
import wandb
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from utils import config
from tqdm import tqdm
from models import resnet
from dataset.dataset import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


class Run():
    
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        global logger
        logger = self.args.logger
        return
    
    
        
    def set_wandb(self):
        wandb.login(key=self.args.wandb_api_key)

        wandb.init(
            project=self.args.project_name,
            name=self.args.experiment_name, 
            job_type=self.args.type,
            config=self.set_wandb_args(),
            notes=f'{self.args.task} task with {self.args.model} model',
            tags=[self.args.task, self.args.model],
            mode=self.args.mode
        )

        return
    
    def set_wandb_args(self):
        d = {
            'seed': self.args.seed,
            'task': self.args.task,
            'model': self.args.resnet,
            'batch_size_val': self.args.batch_size_val,
            'tag': self.args.tag,
            'message': self.args.message
        }

        if self.args.tag == 'train':
            d['batch_size'] = self.args.batch_size
            d['lr'] = self.args.lr
            d['weight_decay'] = self.args.weight_decay
            d['epochs'] = self.args.epochs
        elif self.args.tag != 'eval':
            raise Exception(f'Choose tag flag between train and eval!')

        return d
    
    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def train_loop(self):
        self.model.train()
        train_accuracy = 0
        train_loss = 0
        for i, batch in enumerate(self.train_loader):
            labels, image = batch

            labels = labels.to(self.device)
            image= image.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(image)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            outputs = torch.argmax(outputs, dim=-1)
            acc = accuracy_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            train_accuracy += acc
            train_loss += loss.item()

            

        train_accuracy /= len(self.train_loader)
        train_loss /= len(self.train_loader)
        
        train_metrics = {'train/accuracy': train_accuracy, 'train/loss': train_loss}

        return train_metrics

    def eval_loop(self):
        self.model.eval()
        
        with torch.no_grad():
            eval_accuracy = 0
            eval_loss = 0
            for i, batch in enumerate(self.test_loader):
                labels, image = batch

                labels = labels.to(self.device)
                image= image.to(self.device)
                
                outputs = self.model(image)

                loss = self.criterion(outputs, labels)

                outputs = torch.argmax(outputs, dim=-1)
                acc = accuracy_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                
                eval_accuracy += acc
                eval_loss += loss.item()

            eval_accuracy /= len(self.test_loader)
            eval_loss /= len(self.test_loader)

        eval_metrics = {'eval/accuracy': eval_accuracy, 'eval/loss': eval_loss}

        return eval_metrics

    def train(self):
        '''Train function.'''


        if self.args.wandb:
            # WandB configuration
            self.set_wandb()

        transform = self.get_transform
        train_data = ImageLoader_ResNet(split='train', task=self.args.task, transform=transform)
        test_data = ImageLoader_ResNet(split='test', task=self.args.task, transform=transform)

           

        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.args.batch_size_train,
                                                    shuffle=True, pin_memory=True,
                                                    drop_last=False)
        
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.args.batch_size_val,
                                                    shuffle=False, pin_memory=True,
                                                    drop_last=False)
        

        logger.info('Begin Training -->')
        epochs = self.args.epochs
        self.model = resnet.ResNet(self.args)
        self.model.to(self.device)


        # Define the loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        os.makedirs("checkpoints", exist_ok=True)

        # Training loop
        best_accuracy = 0
        best_loss = 100
        for epoch in tqdm(range(epochs)):
            train_results = self.train_loop()
            eval_results = self.eval_loop()

            metrics = {'train/accuracy': train_results['train/accuracy'], 
                       'eval/accuracy': eval_results['eval/accuracy'], 
                       'train/loss': train_results['train/loss'], 
                       'eval/loss': eval_results['eval/loss']}
            
            if self.args.wandb:
                wandb.log(metrics, step=epoch)
            logger.info(metrics)

            if metrics['eval/accuracy']>best_accuracy:
                best_accuracy = metrics['eval/accuracy']
                best_model = copy.deepcopy(self.model)
                torch.save(best_model.state_dict(), f'checkpoints/{self.args.model}_{self.args.id_run}.pth')
                self.args.checkpoint = f'checkpoints/{self.args.model}_{self.args.id_run}.pth'
            
            if metrics['eval/loss']<best_loss:
                best_loss = metrics['eval/loss']
                
        logger.info(f"Best Accuracy : {best_accuracy} - Best Loss : {best_loss}")
        logger.info("--> End Training!")

        return
    
    def eval(self):
        '''Eval function.'''


        if self.args.wandb:
            # WandB configuration
            self.set_wandb()

        
        transform = self.get_transform
        test_data = ImageLoader_ResNet(split='test', task=self.args.task, transform=transform)
        
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.args.batch_size_val,
                                                    shuffle=False, pin_memory=True,
                                                    drop_last=False)
        

        logger.info('Begin Evaluation -->')
        self.model = resnet.ResNet(self.args)

        if os.path.isfile(self.args.checkpoint):
            checkpoint = self.args.checkpoint
            print("=> loading checkpoint '{}'".format(self.args.checkpoint))
            checkpoint = torch.load(checkpoint)
            self.model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}'".format(self.args.checkpoint))
        else:
            raise Exception(f'Add a valid checkpoint to load in the config file!')
        
        self.model.to(self.device)

        # Define the loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()

        # Training loop

        eval_results = self.eval_loop()

        metrics = {'eval/accuracy': eval_results['eval/accuracy'], 
                    'eval/loss': eval_results['eval/loss']}
        
        if self.args.wandb:
            wandb.log(metrics)
        logger.info(metrics)
                
        logger.info(f"Accuracy : {eval_results['eval/accuracy']} - Loss : {eval_results['eval/loss']}")
        logger.info("--> End Evaluation!")

        return
        



            

    

    
