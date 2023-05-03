import torch
import torch.nn as nn
import torchvision.models as models


class ResClassifier(nn.Module):
    def __init__(self, class_num=14):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc3 = nn.Linear(64, class_num)

    def forward(self, x):
        fc1_emb = self.fc1(x)
        fc2_emb = self.fc2(fc1_emb)
        logit = self.fc3(fc2_emb)
        return logit

class CC_model(nn.Module):
    def __init__(self, num_classes1=14, num_classes2=None):

        if num_classes2 is None:
            num_classes2 = num_classes1

        super(CC_model, self).__init__()
        assert num_classes1 == num_classes2
        self.num_classes = num_classes1
        self.model_resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        num_ftrs = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        self.classification_fc = nn.Linear(num_ftrs, num_classes1)
        self.dr = nn.Linear(num_ftrs, 128)
        self.fc1 = ResClassifier(num_classes1)
        self.fc2 = ResClassifier(num_classes1)

    def forward(self, x, detach_feature=False):

        with torch.no_grad():
            feature = self.model_resnet(x)
            res_out = self.classification_fc(feature)
            if detach_feature:
                feature = feature.detach()
            dr_feature = self.dr(feature)
            # out1 = self.fc1(dr_feature)
            # out2 = self.fc2(dr_feature)
            # output_mean = (out1 + out2) / 2
            # return dr_feature, output_mean

        return dr_feature