import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2



class WiderFaceDetection(data.Dataset):
    def __init__(self, img, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        
        bbox_txt = txt_path + '/list_bbox_celeba.txt'
        landmark_txt = txt_path + '/list_landmarks_celeba.txt'
        identitiy_txt = txt_path + '/identity_CelebA.txt'
        
        f = open(bbox_txt,'r')
        ff = open(landmark_txt, 'r')
        fff = open(identitiy_txt, 'r')
        
        bboxes = f.readlines()
        landmarks = ff.readlines()
        identities = fff.readlines()
        bboxes = bboxes[2:]
        landmarks = landmarks[2:]
        
        for i in range(len(bboxes)):
            labels = []
            bbox = bboxes[i]
            landmark = landmarks[i]
            identity = identities[i]
            bbox = bbox.split(' ')
            bbox = [bb for bb in bbox if bb.strip()]
            landmark = landmark.split(' ')
            landmark = [lan for lan in landmark if lan.strip()]
            identity = identity.split(' ')
            identity = [iden for iden in identity if iden.strip()]
            
            identity = int(identity[1])
            box = [float(x) for x in bbox[1:]]
            land = [float(x) for x in landmark[1:]]
            
            label = box + land
            if identity >= 0:
                label.append(1)
            else:
                label.append(0)
            labels.append(label)
            self.words.append(labels)
            img_path = img + '/' + bbox[0]
            self.imgs_path.append(img_path)
            # if i == 3:
            #     break
            # if line.startswith('#'):
            #     if isFirst is True:
            #         isFirst = False
            #     else:
            #         labels_copy = labels.copy()
            #         self.words.append(labels_copy)
            #         labels.clear()
            #     path = line[2:]
            #     path = txt_path.replace('label.txt','images/') + path
            #     self.imgs_path.append(path)
            # else:
            #     line = line.split(' ')
            #     label = [float(x) for x in line]
            #     labels.append(label)


    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        # img = img.transpose(2, 0, 1)
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225])
        # ])

        # train_transform = transforms.Compose([
        #     transforms.Resize(s),
        #     # transforms.RandomCrop(32, 4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])
        # ])

        # valid_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225])
        # ])

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for label in labels:
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x   lefteye_x
            annotation[0, 5] = label[5]    # l0_y   lefteye_y
            annotation[0, 6] = label[6]    # l1_x   righteye_x
            annotation[0, 7] = label[7]    # l1_y   righteye_y
            annotation[0, 8] = label[8]   # l2_x   nose_x
            annotation[0, 9] = label[9]   # l2_y   nose_y
            annotation[0, 10] = label[10]  # l3_x   leftmouth_x
            annotation[0, 11] = label[11]  # l3_y   leftmouth_y
            annotation[0, 12] = label[12]  # l4_x   rightmouth_x
            annotation[0, 13] = label[13]  # l4_y   rightmouth_y
            annotation[0, 14] = label[14]  # identity
            # if (annotation[0, 4]<0):
            #     annotation[0, 14] = -1
            # else:
            #     annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)


    return (torch.stack(imgs, 0), torch.stack(targets))