import torch
from tqdm.auto import tqdm
import os
from data.datasets import create_datasets, create_data_loaders
from data.preprocess import *




# training
def train(model, priors, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    counter = 0
    
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        # labels = torch.from_numpy([label for label in labels])
        image = image.to(device)
        labels = [label.to(device) for label in labels]
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss_l, loss_c, loss_landm = criterion(outputs, priors, labels)
        # print(loss_l)
        # print(loss_c)
        # print(loss_landm)
        loss = loss_l * 2.0 + loss_c + loss_landm
        train_running_loss += loss.item()
        # calculate the accuracy
        optimizer.zero_grad()
        loss.backward()
        # update the optimizer parameters
        optimizer.step()

    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    # epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    # return epoch_loss, epoch_acc
    return epoch_loss



# validation
def validate(model, priors, testloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = [label.to(device) for label in labels]
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss_l, loss_c, loss_landm = criterion(outputs, priors, labels)
            loss = loss_l * 2.0 + loss_c + loss_landm
            valid_running_loss += loss.item()
            # calculate the accuracy

        
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    # epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    # return epoch_loss, epoch_acc
    return epoch_loss