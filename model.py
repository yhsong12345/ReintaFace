import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from retina.retinaFace import RetinaFace
from data.prior_box import PriorBox
from utils.loss import MultiBoxLoss




class L_RetinaFace(L.LightningModule):
    def __init__(self, m, phase, imgsz, lr, num_classes=10177):
        super().__init__()
        self.lr = lr
        self.model = RetinaFace(m, phase, num_classes)
        self.priors = PriorBox(image_size=(imgsz, imgsz)).forward()
        self.criterion = MultiBoxLoss(num_classes=num_classes, overlap_thresh=0.35, 
                            prior_for_matching=True, bkg_label=0, 
                            neg_mining=True, neg_pos=7, 
                            neg_overlap=0.35, encode_target=False)
        


    def training_step(self, batch, batch_idx):
        # print('Training')
        img, targets = batch
        # print(img.device)
        # print(targets.device)
        prior = self.priors.to(img.device)
        y = self.model(img)
        loss_l, loss_c, loss_landm = self.criterion(y, prior, targets)
        loss = loss_l * 2.0 + loss_c + loss_landm
        values = {'loss_l': loss_l, 'loss_c': loss_c, 'loss_landm': loss_landm, 'Training loss': loss}
        # self.log_dict({"training loss": loss})
        self.log_dict(values, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # print('validation')
        img, targets = batch
        y = self.model(img)
        prior = self.priors.to(img.device)
        loss_l, loss_c, loss_landm = self.criterion(y, prior, targets)
        loss = loss_l * 2.0 + loss_c + loss_landm
        # self.log_dict({"validation loss", loss})
        values = {'loss_l': loss_l, 'loss_c': loss_c, 'loss_landm': loss_landm, 'validation loss': loss}
        self.log_dict(values, prog_bar=True)



    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20, 28],
            gamma=0.1
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    

    # def forward(self, x):
    #     bbox, classification, landmark = self.model(x)
    #     classification = torch.nn.functional.softmax(classification, dim=-1)
    #     return bbox, classification, landmark