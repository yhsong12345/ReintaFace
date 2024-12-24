import torch
import matplotlib.pyplot as plt
import pandas as pd



            

plt.style.use('ggplot')


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

        
    def __call__(
        self, m, current_valid_loss,
        epoch, model, optimizer, criterion
    ):
        path = f'./outputs/{m}'
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")            
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'{path}/best_model.pt')                


def save_model(m, epoch, model, optimizer, criterion):
    print(f'Saving final model...')
    path = f'./outputs/{m}'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        }, f'{path}/model{epoch}.pt')



def save_data(m, train_loss, valid_loss):
    print("Saving losses and accuracies")
    path = f'./outputs/{m}'
    data = {'train_loss': train_loss, 'valid_loss': valid_loss}
    df = pd.DataFrame(data=data)
    df.to_excel(f'{path}/{m}result.xlsx')

    


def save_plots(m, train_loss, valid_loss):
    path = f'./outputs/{m}'

    plt.figure(figsize=(10,7))
    plt.plot(
        train_loss, color='red', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='blue', linestyle='-',
        label='validation loss'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{path}/loss.png')


