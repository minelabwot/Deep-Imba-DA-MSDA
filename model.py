import torch
import torch.nn as nn
import adv_layer
import torch.nn.functional as F
import numpy as np
import resnet

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=20, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(in_features=192, out_features=2, bias=True)

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.pool2(x)

        # x = x.view(-1, 192)  # reshape tensor
        # x = self.fc(x)
        return x




class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        # self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        # 四分类，此处隐去下面一行
        # self.class_classifier.add_module('c_fc1', nn.Linear(10*32,4))
        # self.class_classifier.add_module('c_fc1', nn.Linear(64, 48))
        # self.class_classifier.add_module('c_fc2', nn.Linear(48, 12))
        # self.class_classifier.add_module('c_fc3', nn.Linear(12, 4))
        # self.class_classifier.add_module('c_softmax', nn.Softmax())
        self.fc1 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.fc2 = nn.Linear(in_features=32, out_features=16, bias=True)
        self.fc3 = nn.Linear(in_features=16, out_features=4, bias=True)

    def forward(self, x):
        # x = x.view(-1, 18 * 32)

        # x = x.view(-1,10*32)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.relu(x, inplace=True)
        y = self.fc3(x)
        # add here
        # y = F.softmax(y)
        return y
        # return self.class_classifier(x)



class DANN(nn.Module):

    def __init__(self,device):
        super(DANN, self).__init__()
        self.device = device
        self.feature = FeatureExtractor()
        # self.feature = resnet.resnet34()
        self.classifier = Classifier()
        self.domain_classifier = adv_layer.Discriminator(
            # input_dim=50 * 4 * 4, hidden_dim=100)
            input_dim=64, hidden_dim=32)

    def forward(self, input_data, alpha=1, source=True):
        #print('init',input_data.size())
        # input_data = input_data.expand(len(input_data),20,2)
        #print('here:',input_data.size())
        # feature = self.feature(input_data).unsqueeze(2)
        feature = self.feature(input_data)
        # print('log-feature-size', feature.size())
        tem_feature = feature
        # np.save('/home/zhk/transferLearning/bearing_cppy/bearing/DANN-tr/temp_feature'+'/temp_feature.npy',feature.detach())
        # print('log66',feature.size())
        feature = feature.view(-1, 64)
        # feature = feature.unsqueeze(2)
        # print('log67',feature.size())
        # print('log67',feature.size())
        #此处隐去下面一行
        # feature = feature.view(-1,10*32)
        # print('log7', feature.size())
        class_output = self.classifier(feature)
        # print('log69',class_output)
        domain_output,domain_pred = self.get_adversarial_result(
            feature, source, alpha)
        # print('log102',domain_output)
        return class_output, domain_output,tem_feature,domain_pred

    def get_adversarial_result(self, x, source=True, alpha=1):
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(self.device)
        else:
            domain_label = torch.zeros(len(x)).long().to(self.device)
        x = adv_layer.ReverseLayerF.apply(x, alpha)
        # print('log66')
        # print('log8',x.size())
        domain_pred = self.domain_classifier(x)
        # domain_pred 输出的是0到1之间的概率
        # print('log00000',domain_pred.size())
        # domain_pred[domain_pred<0] = 0.0
        # domain_pred[domain_pred>1] = 1.0
        loss_adv = loss_fn(domain_pred, domain_label.float())
        return loss_adv, domain_pred
