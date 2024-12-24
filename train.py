import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from plots import *
from data.datasets import create_datasets, create_data_loaders
from utils.loss import MultiBoxLoss
import os
from utils.engine import *
from retina.retinaFace import RetinaFace
from data.prior_box import PriorBox
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor

from model import L_RetinaFace



# try:
#     rank = int(os.environ["RANK"])
#     local_rank = int(os.environ['LOCAL_RANK'])
#     world_size = int(os.environ['WORLD_SIZE'])
#     print(rank, local_rank, world_size)
#     torch.distributed.init_process_group('nccl')
# except KeyError:
#     rank = 0
#     local_rank = 0
#     world_size = 1
#     torch.distributed.init_process_group(
#         backend='nccl',
#         init_method='tcp://127.0.0.1:12584',
#         rank=rank,
#         world_size=world_size
#         )


def main(args):
# build the model
# learning_parameters 
    lr = args['learning_rate']
    epochs = args['epochs']
    BATCH_SIZE = args['batch_size']
    s = args['image_size']
    m = args['model']
    d = args['save_dir']
    f = args['data']
    
    save_dir = os.getcwd() + '/' + d + f"/{m}"

    # Check the save_dir exists or not
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    min_epochs = min(100, epochs)
    max_epochs = max(epochs, 300)

    train_dataset, valid_dataset, _ = create_datasets(s, f)
    # get the training and validaion data loaders
    train_loader, valid_loader = create_data_loaders(
        train_dataset, valid_dataset, BATCH_SIZE
    )

    wandblogger = WandbLogger(project='face recognition', name='retinaface', log_model='all')

    model = L_RetinaFace(m, phase='train', imgsz=s, lr=lr, num_classes=2)
    trainer = L.Trainer(default_root_dir=save_dir, accelerator='gpu', min_epochs=min_epochs, max_epochs=max_epochs, 
                        profiler='simple', devices=2,
                        num_sanity_val_steps=0,
                        callbacks=[ModelCheckpoint(save_top_k=3, monitor='validation loss', mode='min'),
                                   ModelSummary(max_depth=3),
                                  LearningRateMonitor(logging_interval='epoch')],
                                   logger=wandblogger)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)






# def main(args):
# # build the model
# # learning_parameters 
#     lr = args['learning_rate']
#     epochs = args['epochs']
#     BATCH_SIZE = args['batch_size']
#     s = args['image_size']
#     m = args['model']
#     d = args['save_dir']
#     f = args['data']
#     # local_rank = args['local_rank']

#     # computation device
#     device = ('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Computation device: {device}\n")


#     # get the training, validation and test_datasets
#     train_dataset, valid_dataset, _ = create_datasets(s, f)
#     # get the training and validaion data loaders
#     train_loader, valid_loader = create_data_loaders(
#         train_dataset, valid_dataset, BATCH_SIZE
#     )


#     save_dir = d + f"/{m}"

#     # Check the save_dir exists or not
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     Model = RetinaFace(m, phase='train')
#     model = Model.to(device)
#     print(model)
#     # if device and torch.cuda.device_count() > 1:
#     #     model = nn.DataParallel(model)
    
#     ## Multiple GPUs
#     # model = torch.nn.parallel.DistributedDataParallel(
#     #     module=model, broadcast_buffers=False, device_ids=[local_rank], 
#     #     bucket_cap_mb=16, find_unused_parameters=True
#     # )
#     # model.register_comm_hook(None, fp16_compress_hook)

#     priors = PriorBox(image_size = (s, s))
#     with torch.no_grad():
#         priors = priors.forward()
#         priors = priors.to(device)
#     # priors = torch.nn.parallel.DistributedDataParallel(
#     #     module=priors, broadcast_buffers=False, device_ids=[local_rank], 
#     #     bucket_cap_mb=16, find_unused_parameters=True
#     #     )
#     # priors.register_comm_hook(None, fp16_compress_hook)


#     # total parameters and trainable parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"{total_params:,} total parameters.")
#     total_trainable_params = sum(
#         p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"{total_trainable_params:,} training parameters.\n")


#     # loss function
#     # criterion = nn.CrossEntropyLoss()
#     criterion = MultiBoxLoss(num_classes=2, overlap_thresh=0.35, 
#                             prior_for_matching=True, bkg_label=0, 
#                             neg_mining=True, neg_pos=7, 
#                             neg_overlap=0.35, encode_target=False)

#     # if h:
#     #     model.half()
#     #     criterion.half()

#     # optimizer
#     optimizer = optim.SGD(model.parameters(), lr=lr,
#                         momentum=0.9, weight_decay=5e-4)

#     lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
#                                                 milestones=[20, 28], 
#                                                 gamma=0.1)


#     # initialize SaveBestModel class
#     save_best_model = SaveBestModel()


#     # lists to keep track of losses and accuracies
#     train_loss, valid_loss = [], []

#     # start the training
#     for epoch in range(epochs):
#         print(f"[INFO]: Epoch {epoch+1} of {epochs}")
#         train_epoch_loss = train(model, priors, train_loader, 
#                                                 optimizer, criterion, device)
        
#         valid_epoch_loss = validate(model, priors, valid_loader,  
#                                                     criterion, device)

#         lr_scheduler.step()
#         train_loss.append(train_epoch_loss)
#         valid_loss.append(valid_epoch_loss)
#         # train_acc.append(train_epoch_acc)
#         # valid_acc.append(valid_epoch_acc)
#         print(f"Training loss: {train_epoch_loss:.3f}")
#         print(f"Validation loss: {valid_epoch_loss:.3f}")
#         # print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
#         # print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
#         save_model(m, epoch, model, optimizer, criterion)
#         # save the best model till now if we have the least loss in the current epoch
#         save_best_model(
#             m, valid_epoch_loss, epoch, model, optimizer, criterion
#         )
#         print('-'*50)
        
#     # save the trained model weights for a final time
#     # save_model(m, epochs, model, optimizer, criterion)
#     save_data(m, train_loss, valid_loss)
#     # save the loss and accuracy plots
#     save_plots(m, train_loss, valid_loss)
#     print('TRAINING COMPLETE')

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument('-e', '--epochs', type=int, default=300, help='number of epochs to train our network for')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('-s', '--image_size', type=int, default=640, help='image size')
    parser.add_argument('-m', '--model', type=str, default= 'mobilenet0.25', help='Model Selection')
    parser.add_argument('-d', '-sav_dir', type=str, dest='save_dir', help='directory', default='outputs')
    parser.add_argument('--data', type=str, default='./data/celeba')
    parser.add_argument('-r', '--local_rank', type=int, default=0, help='number of gpus')
    args = vars(parser.parse_args())
    main(args)