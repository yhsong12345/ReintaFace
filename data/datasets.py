import torch
from torch.utils.data import dataloader, Subset, distributed, DataLoader
from data.preprocess import WiderFaceDetection, detection_collate
from data.data_augment import preproc
import torchvision.transforms as transforms
import torchvision
import os
import numpy as np
import random




def create_datasets(s, f):
    path = f.replace('.',f'{os.getcwd()}')
    img = path + '/Img/img_celeba.7z/img_celeba'
    Annotation = path + '/Anno'
    evaluation = path + '/Eval/list_eval_partition.txt'
    dataset = WiderFaceDetection(img, Annotation, preproc=preproc(img_dim = s, rgb_means = (104, 117, 123)))
    indices = divide_dataset(evaluation)
    train_dataset = Subset(dataset, list(range(indices[0], indices[1])))
    valid_dataset = Subset(dataset, list(range(indices[1], indices[2])))
    test_dataset = Subset(dataset, list(range(indices[2], len(dataset))))
    return train_dataset, valid_dataset, test_dataset


# def create_datasets(s):
#     train_transform = transforms.Compose([
#         transforms.Resize(s),
#         # transforms.RandomCrop(32, 4),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     valid_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
#     ])

#     train_dataset = torchvision.datasets.CelebA(root='.\\data', split='train', download=True, transform=train_transform)
#     val_dataset = torchvision.datasets.CelebA(root='.\\data', split='val', download=True, transform=valid_transform)

#     return train_dataset, val_dataset


def divide_dataset(evaluation):
    f = open(evaluation, 'r')
    lines = f.readlines()
    boundary = []
    prev = 0
    for i in range(len(lines)):
        line = lines[i]
        line = line.split(' ')
        line = [i for i in line if i.strip()]
        ty = line[-1]
        ty = ty.replace('\n', '')
        if prev != ty:
            boundary.append(i)
            prev = ty

    return boundary




def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


def create_data_loaders(dataset_train, dataset_valid, BATCH_SIZE):
    """
    Function to build the data loaders.
    Parameters:
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    :param dataset_test: The test dataset.
    """
    rank = int(os.getenv("RANK", -1))
    nd = torch.cuda.device_count()
    train_shuffle=True
    val_shuffle=False
    # train_sampler = None if rank == -1 else distributed.DistributedSampler(dataset_train, shuffle=train_shuffle)
    # val_sampler = None if rank == -1 else distributed.DistributedSampler(dataset_valid, shuffle=val_shuffle)
    nw = os.cpu_count() // max(nd, 1)  # number of workers
    nw = min(8, nw)
    
#     generator = torch.Generator()
#     generator.manual_seed(6148914691236517205 + rank)
    
    # print(f"number of workers: {nw}")
    
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=nw, collate_fn = detection_collate
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, collate_fn = detection_collate
    )
    
    
    # train_loader = DataLoader(
    #     dataset_train, batch_size=BATCH_SIZE, shuffle=train_shuffle and train_sampler is None, num_workers=nw, collate_fn = detection_collate, generator=generator, worker_init_fn=seed_worker, pin_memory=PIN_MEMORY
    # )
    # valid_loader = DataLoader(
    #     dataset_valid, batch_size=BATCH_SIZE, shuffle=val_shuffle and val_sampler is None, num_workers=nw, collate_fn = detection_collate, generator=generator, worker_init_fn=seed_worker, pin_memory=PIN_MEMORY
    # )
    return train_loader, valid_loader



# class InfiniteDataLoader(dataloader.DataLoader):
#     """
#     Dataloader that reuses workers.

#     Uses same syntax as vanilla DataLoader.
#     """

#     def __init__(self, *args, **kwargs):
#         """Dataloader that infinitely recycles workers, inherits from DataLoader."""
#         super().__init__(*args, **kwargs)
#         object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
#         self.iterator = super().__iter__()

#     def __len__(self):
#         """Returns the length of the batch sampler's sampler."""
#         return len(self.batch_sampler.sampler)

#     def __iter__(self):
#         """Creates a sampler that repeats indefinitely."""
#         for _ in range(len(self)):
#             yield next(self.iterator)

#     def reset(self):
#         """
#         Reset iterator.

#         This is useful when we want to modify settings of dataset while training.
#         """
#         self.iterator = self._get_iterator()



# class _RepeatSampler:
#     """
#     Sampler that repeats forever.

#     Args:
#         sampler (Dataset.sampler): The sampler to repeat.
#     """

#     def __init__(self, sampler):
#         """Initializes an object that repeats a given sampler indefinitely."""
#         self.sampler = sampler

#     def __iter__(self):
#         """Iterates over the 'sampler' and yields its contents."""
#         while True:
#             yield from iter(self.sampler)