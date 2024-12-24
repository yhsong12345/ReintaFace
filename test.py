import torch
from tqdm.auto import tqdm
from plots import *
import argparse
from model import *
from datasets import create_datasets, create_data_loaders
# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=200,
    help='number of epochs to train our network for')
parser.add_argument('-lr', '--learning_rate', type=float, 
                    default=0.1, help='learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=128,
                    help='Batch Size')
parser.add_argument('-s', '--image_size', type=int, default=32,
                    help='image size')
parser.add_argument('-m', '--model', type=str, default= 'Resnet20',
                    help='Model Selection')
parser.add_argument('-d', '-sav_dir', type=str, dest='save_dir', help='directory', default='outputs')
args = vars(parser.parse_args())

# learning_parameters 
lr = args['learning_rate']
epochs = args['epochs']
BATCH_SIZE = args['batch_size']
s = args['image_size']
m = args['model']
d = args['save_dir']
h = True

# build the model, no need to load the pre-trained weights or fine-tune layers
model = SelectModel(m).to(device)
# load the best model checkpoint
best_model_cp = torch.load(f'outputs/{m}/best_model.pt')
best_model_epoch = best_model_cp['epoch']
print(f"Best model was saved at {best_model_epoch} epochs\n")
# # load the last model checkpoint
# last_model_cp = torch.load(f'outputs/{m}/final_model.pt')
# last_model_epoch = last_model_cp['epoch']
# print(f"Last model was saved at {last_model_epoch} epochs\n")
# get the test dataset and the test data loader
train_dataset, valid_dataset, test_dataset = create_datasets(s)
_, _, test_loader = create_data_loaders(
    train_dataset, valid_dataset, test_dataset, BATCH_SIZE)



def test(model, testloader):
    """
    Function to test the model
    """
    # set model to evaluation mode
    model.eval()
    print('Testing')
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    final_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return final_acc


# test the best epoch saved model
def test_best_model(model, checkpoint, test_loader):
    print('Loading best epoch saved model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_acc = test(model, test_loader)
    print(f"Best epoch saved model accuracy: {test_acc:.3f}")


if __name__ == '__main__':
    # test_last_model(model, last_model_cp, test_loader)
    test_best_model(model, best_model_cp, test_loader)