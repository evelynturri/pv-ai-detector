import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):

    def __init__(self, args) -> None:
        super(ResNet, self).__init__()
        #Define the pretrained architecture
        self.resnet = self.get_resnet_model(args.resnet)

        if args.transfer_learning:
            # Freeze the parameters of the original ResNet model
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Save the output size of the last layer because we will need it later
        num_features = self.resnet.fc.in_features

        # Replace it with a new layer for your task
        self.resnet.fc = nn.Linear(num_features, args.num_classes)

        # Unfreeze the last layer so that it can be trained
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    
    def get_resnet_model(self, model: str = 'resnet50', *args, **kwargs) -> nn.Module:
        if model == 'resnet50':
            return models.resnet50(weights="IMAGENET1K_V2")
        elif model == 'resnet101':
            return models.resnet101(weights="IMAGENET1K_V2")
        elif model == 'resnet18':
            return models.resnet18(weights="IMAGENET1K_V1")
        else:
            raise Exception(f"Model {model} not implemented.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We have to squeeze the output since it is a 4D tensor and we want a 2D one
        x = self.resnet(x).squeeze().squeeze()

         # Apply softmax to get probabilities for each class
        x = torch.softmax(x, dim=-1)
        
        return x

