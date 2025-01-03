import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F

from retina.net import MobileNetV1 as MobileNetV1
from retina.net import FPN as FPN
from retina.net import SSH as SSH
from utils.loss import ArcFace



########################## original ################################
class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3, num_classes=2):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*num_classes, kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, self.num_classes)

class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,  num_anchors*4, kernel_size=(1,1), stride=1, padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*10, kernel_size=(1,1), stride=1, padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)
    

    


class RetinaFace(nn.Module):
    def __init__(self, m, phase, num_classes):

        super().__init__()
        backbone = None
        self.phase = phase
        if m == 'mobilenet0.25':
            backbone = MobileNetV1()
            return_layers = {'stage1': 1, 'stage2': 2, 'stage3': 3}
            in_channel =  32
            out_channel = 64

        elif m == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=False)
            return_layers = {'layer2': 1, 'layer3': 2, 'layer4': 3}
            in_channel = 256
            out_channel = 256

        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
        in_channels_stage2 = in_channel
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = out_channel
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=out_channel, num_classes=num_classes)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=out_channel)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=out_channel)

    def _make_class_head(self,fpn_num=3,inchannels=64, anchor_num=2, num_classes=2):
        classhead = nn.ModuleList()
        for _ in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num, num_classes))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for _ in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for _ in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead
    
    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        # if self.phase == 'train':
        #     output = (bbox_regressions, classifications, ldm_regressions)
        # else:
        #     output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
    
        return bbox_regressions, classifications, ldm_regressions