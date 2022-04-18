import numpy as np
import data_loader_smote
import torch
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import tqdm
import data_loader
from model import DANN
import os
import argparse
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn import metrics
import plot_lstm_feature
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler
import time
import center_loss
import json
from pytorchtools import EarlyStopping
from collections import Counter
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=.5)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lr_cent', type=float, default=0.5)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--nepoch', type=int, default=100)
parser.add_argument('--source', type=str, default=0)
parser.add_argument('--target', type=str, default=3)
parser.add_argument('--model_path', type=str, default='models')
parser.add_argument('--result_path', type=str, default='/result.csv')
parser.add_argument('--factor', type=float, default=.1)
parser.add_argument('--mmd_weight', type=float, default=.5)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--version', type=str, default='unknown')
parser.add_argument('--finetue',type=str,default='true')
parser.add_argument('--detail',type=str,default='with imbalance')
args = parser.parse_args()

DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')


# 绘制伪标签各分类的指标
def plot_for_pesudo(r0, r1, r2, r3, path, filename, epochs=args.nepoch):
    x = range(epochs)
    total = [r0, r1, r2, r3]
    for i in range(1, 5):
        plt.subplot(4, 1, i)
        plt.plot(x, total[i - 1], '.-')
        plt.ylabel('class {}'.format(i - 1))
    # plt.title('precision of different class  vs epoches')
    plt.savefig('{}/{}_different_class.jpg'.format(path, filename))
    plt.close()


# 高斯核
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total = total.view(int(total.size(0)), -1)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # nan_mask = torch.tensor([])
    # for bandwidth_temp in bandwidth_list:
    #     nan_mask.cat(torch.isnan(-L2_distance / bandwidth_temp))
    # kernel_val = [torch.exp((-L2_distance / bandwidth_temp)[~nan_mask]) for bandwidth_temp in bandwidth_list]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


# 等长，没有计算均值

def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss = loss + kernels[s1, s2] + kernels[t1, t2]
        loss = loss - (kernels[s1, t2] + kernels[s2, t1])
    return loss  # / float(batch_size)


def mmd_rbf_noaccelerate(source, target, sample_weight, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size].mul(sample_weight)   #bug
    YY = kernels[batch_size:, batch_size:].mul(sample_weight)
    XY = kernels[:batch_size, batch_size:].mul(sample_weight)
    YX = kernels[batch_size:, :batch_size].mul(sample_weight)
    loss = torch.mean(XX + YY - XY - YX)
    print('XX_size', XX.size())
    return loss


# def mmd_rbf_noaccelerate_ori(source, target, prob_pesudo, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#     batch_size = int(source.size()[0])
#     kernels = guassian_kernel(source, target,
#                               kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
#     XX = kernels[:batch_size, :batch_size]
#     YY = kernels[batch_size:, batch_size:]*(prob_pesudo**2)   //py 目标域打上该类别标签的样本数目/batch_size
#     XY = kernels[:batch_size, batch_size:]*prob_pesudo
#     YX = kernels[batch_size:, :batch_size]*prob_pesudo     //.mul() 样本级别， softmax
#     loss = torch.mean(XX + YY - XY - YX)
#     print('XX_size', XX.size())
#     return loss

# def mmd_rbf_noaccelerate_ori(source, target, prob_pesudo, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#     batch_size = int(source.size()[0])
#     kernels = guassian_kernel(source, target,
#                               kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
#     XX = kernels[:batch_size, :batch_size]
#     YY = kernels[batch_size:, batch_size:]
#     XY = kernels[:batch_size, batch_size:]
#     YX = kernels[batch_size:, :batch_size]
#     loss = torch.mean(XX + YY - XY - YX) * prob_pesudo
#     print('XX_size', XX.size())
#     return loss

# prob_pesudo 对应伪标签softmax的输出
def mmd_rbf_noaccelerate_ori(source, target, prob_pesudo, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    prob_pesudo = F.softmax(prob_pesudo)
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:].mul(prob_pesudo**2)
    XY = kernels[:batch_size, batch_size:].mul(prob_pesudo)
    YX = kernels[batch_size:, :batch_size].mul(prob_pesudo)
    loss = torch.mean(XX + YY - XY - YX)
    # print('XX_size', XX.size())
    return loss


# prob1 源域某种样本的概率，prob2 目标域伪标签下某种样本的概率
def mmd_rbf_accelerate_v1(source, target, prob1, prob2, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss = loss + prob1 * prob1 * kernels[s1, s2] + prob2 * prob2 * kernels[t1, t2]
        loss = loss - prob1 * prob2 * kernels[s1, t2] - prob1 * prob2 * kernels[s2, t1]
    return loss  # / float(batch_size)


# 划分类内，计算mmd
# def intra_MMD():

def my_sigmoid(X):
    return 1.0 / (1 + np.exp(-X))


def scale(X):
    # X = X.detach().numpy()
    # return -4*np.power((X-0.5), 2)+1
    return np.exp(-np.abs(X - 0.5))
    # return torch.exp(-torch.abs(X-0.5))


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def cal_real_class_weight(label_path):
    train_label = np.load(label_path)
    dic = Counter(train_label.flatten())
    weight = []
    for i in range(4):
        weight.append(dic.get(i) / train_label.shape[0])
    return weight


def cal_class_weight(label_path):
    train_label = np.load(label_path)
    one_hot_encoder = preprocessing.OneHotEncoder(categories="auto")
    one_hot_encoder.fit(train_label)
    train_label_encoder = one_hot_encoder.transform(train_label).toarray()
    y_integers = np.argmax(train_label_encoder, axis=1)
    class_weights = class_weight.compute_class_weight("balanced",
                                                      np.unique(y_integers),
                                                      y_integers)
    return class_weights


def cal_prob(label_path):
    train_label = np.load(label_path)
    one_hot_encoder = preprocessing.OneHotEncoder(categories="auto")
    one_hot_encoder.fit(train_label)
    train_label_encoder = one_hot_encoder.transform(train_label).toarray()
    y_integers = np.argmax(train_label_encoder, axis=1)
    prob = []
    for i in range(4):
        prob.append(len(np.where(y_integers == i)[0]) / len(y_integers))

    return prob


def test(model, flag, epoch, src, tar, batch_size):
    alpha = 0
    # dataloader = data_loader.load_test_data(dataset_name)
    if flag == 0:
        dataloader, dataloader1 = data_loader.load_data(src=src, tar=tar, batch_size=batch_size)
    else:
        dataloader = data_loader.load_test_data(tar=tar, batch_size=batch_size)
    model.eval()
    # n_correct = 0
    n_correct = 0
    # n_correct_fault1 =0
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(dataloader):
            t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE)
            t_img = t_img.float()
            class_output, _, _, _ = model(input_data=t_img, alpha=alpha)
            # print('log55',class_output.size())

            prob, pred = torch.max(class_output.data, 1)

            t_label = t_label.squeeze(1)
            n_correct += (pred == t_label.long()).sum().item()

    acc = float(n_correct) / len(dataloader.dataset) * 100

    return acc


def train(model, optimizer, dataloader_src, dataloader_tar, patience):
    dic_path = "saved_model/{}_DANN_sliding_{}_motor_train_{}_test_{}".format(
        time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
        20, args.source, args.target)
    mkdir(dic_path)
    # 计算权重
    weight = cal_class_weight(
        '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_{}_label.npy'.format(args.source))
    weight = torch.from_numpy(weight).float()
    # 均衡比，类别数目/总数目
    real_weight = cal_real_class_weight('/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_{'
                                        '}_label.npy'.format(args.source))
    real_weight = torch.from_numpy(np.array(real_weight)).float()
    # 源域分布
    prob1 = cal_prob(
        '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_{}_label.npy'.format(args.source))

    loss_class = torch.nn.CrossEntropyLoss(weight=weight).to(DEVICE)
    # loss_class = torch.nn.CrossEntropyLoss()
    best_acc = -float('inf')
    len_dataloader = min(len(dataloader_src), len(dataloader_tar))
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    mmd_loss_list = []
    acc_0 = []
    acc_1 = []
    acc_2 = []
    acc_3 = []
    recall_0 = []
    recall_1 = []
    recall_2 = []
    recall_3 = []

    valid_losses = []
    valid_loss_epoch = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    print('log99', len(dataloader_src.dataset))
    optimizer_f = optim.SGD(list(model.feature.parameters()), lr=args.lr)
    optimizer_c = optim.SGD(list(model.classifier.parameters()) + list(model.domain_classifier.parameters()),
                            lr=args.lr)
    for epoch in range(args.nepoch):
        adjust_learning_rate(optimizer, epoch)
        print('learning-rate', optimizer.param_groups[0]['lr'])
        with torch.autograd.set_detect_anomaly(True):
            model.train()
            i = 1
            total_classout_tar = torch.empty(0, 4).cuda()
            # 添加 查看源域数据集上的分类表现
            total_classout_src = torch.empty(0, 4).cuda()
            total_tar_y = torch.empty(0).long().cuda()
            total_src_y = torch.empty(0).long().cuda()
            total_domain_pred_src = torch.empty(0).float().cuda()
            # 拼接特征,此处为了调通，不灵活
            total_feature_src = torch.empty(0, 64, 1).cuda()
            total_featrue_tar = torch.empty(0, 64, 1).cuda()
            # 额外保存 epoch 0 和1 的特征 和目标域的预测标签
            total_feature_src0 = torch.empty(0, 64, 1).cuda()
            total_featrue_tar0 = torch.empty(0, 64, 1).cuda()
            total_feature_src1 = torch.empty(0, 64, 1).cuda()
            total_featrue_tar1 = torch.empty(0, 64, 1).cuda()
            total_classout_tar0 = torch.empty(0, 4).cuda()
            total_classout_tar1 = torch.empty(0,4).cuda()
            for (data_src, data_tar) in tqdm.tqdm(zip(enumerate(dataloader_src), enumerate(dataloader_tar)),
                                                  total=len_dataloader, leave=False):
                _, (x_src, y_src) = data_src
                _, (x_tar, y_tar) = data_tar
                x_src, y_src, x_tar = x_src.to(
                    DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE)
                y_tar = y_tar.to(DEVICE)
                p = float(i + epoch * len_dataloader) / args.nepoch / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                x_src = x_src.float()
                x_tar = x_tar.float()
                class_output, err_s_domain, src_temp_feature, domain_pred_src = model(input_data=x_src, alpha=alpha)
                y_src = y_src.long()
                y_tar = y_tar.long()
                # print('log_loss_class',y_src.size())
                y_src = y_src.squeeze(1)
                # 压缩处理
                domain_pred_src = domain_pred_src.squeeze(1)

                err_s_label = loss_class(class_output, y_src)

                class_output_tar, err_t_domain, tar_temp_feature, domain_pred_tar = model(
                    input_data=x_tar, alpha=alpha, source=False)
                # 计算p（y）

                #                              计算类内mmd
                #   划分类内特征
                # mmd_inner = 0
                # mmd_inner = torch.tensor(0.0, requires_grad=True)
                # 变量转numpy
                domain_pred_tar = domain_pred_tar.cpu().detach().numpy()
                mmd_inner = 0
                # 累积梯度，参数一次更新
                # optimizer_f.zero_grad()
                # |domain_pred-0.5| 归一化，定义 μ2
                domain_pred_src_mean_abs = torch.abs(torch.mean(domain_pred_src) - 0.5)
                # print("domain_pred_src_mean_abs:", domain_pred_src_mean_abs)
                # mmd_inner = mmd_rbf_noaccelerate_ori(src_temp_feature, tar_temp_feature)
                # inner mmd start
                for j in range(4):
                    # 源域特征
                    # temp_f_src = src_temp_feature[torch.where(y_src == j)]
                    # 源域特征，非对应标签，替换成零张量
                    temp_f_src = torch.where(y_src == j, src_temp_feature,
                                             torch.zeros(src_temp_feature[0].size()).cuda())
                    # print('temp_src_feature_size', temp_f_src.size())
                    # print('domain_pred_src_size', domain_pred_src.size())
                    #           样本级别权重注入, 作用于每个核值
                    temp_src_factor = torch.where(y_src == j,
                                                  1 / (torch.max(domain_pred_src) - torch.min(domain_pred_src)) *
                                                  torch.abs(domain_pred_src - 0.5),
                                                  torch.zeros(domain_pred_src[0].size()).cuda())
                    # print('temp_src_factor', temp_src_factor)
                    # temp_f_src = temp_f_src.mul(1-domain_pred_src)

                    # temp_index = torch.where(torch.max(class_output_tar.data,1)[1] == j)
                    # print('grad_temp_f_src', temp_f_src.requires_grad)
                    temp_index = np.where((torch.max(class_output_tar.data, 1)[1] == j).cpu().detach().numpy())
                    # print("temp_index", temp_index.)
                    # print(len(temp_index[0])/args.batchsize)
                    # pesudo_prob = len(temp_index[0])/args.batchsize;
                    # pesudo_prob 伪标签的softmax
                    # pesudo_prob = torch.where(y_tar == j, class_output_tar, torch.zeros(class_output_tar[0].size()).cuda())
                    # pesudo_prob = torch.max(pesudo_prob,1)[0].unsqueeze(1)
                    # pesudo_prob = torch.where(torch.max(class_output_tar,1)[1].unsqueeze(1) == j, class_output_tar, torch.zeros(class_output_tar[0].size()).cuda())
                    # pesudo_prob = F.softmax(class_output_tar)
                    pesudo_prob = class_output_tar
                    pesudo_prob = torch.max(pesudo_prob, 1)[0].unsqueeze(1)
                    # print("pesudo_prob_size",pesudo_prob.size())
                    # print('pesudo_prob',pesudo_prob)

                    # print('type_temp_index[0]',type(temp_index[0]))
                    # print(temp_index)
                    # 目标域特征
                    # temp_f_tar = x_tar[temp_index[0]]
                    # temp_f_tar = tar_temp_feature[temp_index]
                    temp_f_tar = torch.where(torch.max(class_output_tar.data, 1)[1] == j, tar_temp_feature,
                                             torch.zeros(64, 1).cuda())
                    # print('grad_temp_f_tar', temp_f_tar.requires_grad)
                    # 计算 目标域标签分布概率
                    prob2 = temp_f_tar.size()[0] / args.batchsize
                    # print("the src shape",temp_f_src.shape)
                    # print("the shape", temp_f_tar.shape)
                    # 域权重因子，使用均值
                    # print("j",j)
                    # print("domain_pred_tar",type(domain_pred_tar))

                    # print(type(domain_pred_tar))
                    # scale_inner = domain_pred_tar / (1 - domain_pred_tar)
                    scale_inner = domain_pred_tar
                    # print("type_scale_inner",type(scale_inner))
                    temp_domain_pred_tar = scale(scale_inner)[temp_index[0]]
                    # temp_domain_pred_tar = scale(scale_inner)[temp_index]
                    # print('ballalal')
                    temp_tar_factor = np.nanmean(temp_domain_pred_tar)
                    # print("temp_tar_factor", temp_tar_factor)
                    # temp_tar_factor = torch.mean(temp_domain_pred_tar).item()
                    # print('type of temp_tar_factor', type(temp_tar_factor))
                    # print('tem_tar_factor_grad',temp_tar_factor.requires_grad)
                    # print(temp_tar_factor)
                    # temp_tar_factor = np.mean(scale(domain_pred_tar/(1-domain_pred_tar))[temp_index])
                    # print('i am here')
                    #  mmd_inner, 不引入非均衡权重
                    # print('temp_tar_factor',temp_tar_factor)
                    # if(np.isnan(temp_tar_factor)):
                    #     temp_tar_factor = 0.5
                    if np.isnan(temp_tar_factor):
                        # print('here nan exist in temp_tar_factor')
                        # temp_tar_factor = 0.5
                        continue
                    # single_mmd = mmd_rbf_accelerate(temp_f_src, temp_f_tar) * temp_tar_factor
                    # print('single_mmd', single_mmd)
                    # optimizer_f.zero_grad()
                    # single_mmd.backward(retain_graph=True)
                    # torch.nn.utils.clip_grad_norm_(list(model.feature.parameters()), max_norm=20, norm_type=2)
                    # optimizer_f.step()
                    # mmd_inner = mmd_inner + single_mmd.item()
                    # print('before_single_mmd', mmd_rbf_noaccelerate(temp_f_src, temp_f_tar, temp_src_factor))
                    # mmd inner 引入非均衡权重
                    # single_mmd = mmd_rbf_noaccelerate_ori(temp_f_src, temp_f_tar, temp_src_factor) * args.mmd_weight
                    # single_mmd = mmd_rbf_noaccelerate_ori(temp_f_src, temp_f_tar) * 1 / real_weight[j] * args.mmd_weight
                    single_mmd = mmd_rbf_noaccelerate_ori(temp_f_src, temp_f_tar, pesudo_prob)  * args.mmd_weight
                    # print('single_mmd', single_mmd)
                    # 2 optimizer_f.zero_grad()
                    # single_mmd.backward(retain_graph=True)
                    # 之前max_norm为20
                    torch.nn.utils.clip_grad_norm_(list(model.feature.parameters()), max_norm=40, norm_type=2)
                    # 2 optimizer_f.step()
                    # mmd_inner = mmd_inner + single_mmd.item()
                    mmd_inner = mmd_inner + single_mmd
                    # print('mmd_requires_grad', mmd_inner.requires_grad)
                # inner mmd end
                # optimizer_f.step()
                # mmd_inner += mmd_rbf_accelerate(temp_f_src, temp_f_tar).item() * temp_tar_factor
                # mmd_inner, 引入非均衡权重
                # mmd_inner += mmd_rbf_accelerate(temp_f_src, temp_f_tar).item() * temp_tar_factor * weight[j]
                # mmd_inner mmd_cda
                # print("only mmd grad", mmd_rbf_accelerate_v1(temp_f_src, temp_f_tar, prob1=prob1[j],
                #                                              prob2=prob2).requires_grad)
                # print('type only mmd', type(mmd_rbf_accelerate_v1(temp_f_src, temp_f_tar, prob1=prob1[j],
                #                                                   prob2=prob2)))
                # single_mmd_inner = mmd_rbf_accelerate_v1(temp_f_src, temp_f_tar, prob1=prob1[j],
                #                                          prob2=prob2) * temp_tar_factor
                # print('single', single_mmd_inner.requires_grad)

                # optimizer_f.zero_grad()
                # single_mmd_inner.backward(retain_graph=True)
                # single_mmd_inner.backward()
                # optimizer_f.step()

                # print("mmd_inner.requires_grad", mmd_inner.requires_grad)
                # print('type of mmd_inner',type(mmd_inner))
                #  mmd_cda,引入非均衡权重
                # mmd_inner += mmd_rbf_accelerate_v1(temp_f_src, temp_f_tar, prob1=prob1[j],
                #                                    prob2=prob2).item() * temp_tar_factor * weight[j]
                # print('log_mmd_inner', mmd_inner)
                # mmd_inner = mmd_inner.item()
                # print('mmd_inner', mmd_inner.requires_grad)
                # !!!!!!!!

                err_domain = err_t_domain + err_s_domain
                # print('log_err_domain', err_domain)
                # err = err_s_label + args.gamma * err_domain + args.factor * center
                #                             定义loss
                err = err_s_label + args.gamma * err_domain + mmd_inner  # mmd_inner 前面的权重需要考虑
                # err = err_s_label + mmd_inner
                # err = err_s_label + (1 - domain_pred_src_mean_abs) * err_domain + mmd_inner * domain_pred_src_mean_abs
                # 拼接class_output,
                total_classout_tar = torch.cat((total_classout_tar, class_output_tar), 0)
                total_classout_src = torch.cat((total_classout_src, class_output), 0)
                total_tar_y = torch.cat((total_tar_y, y_tar), 0)
                # 还原
                y_src = y_src.reshape(-1, 1)
                total_src_y = torch.cat((total_src_y, y_src), 0)
                total_feature_src = torch.cat((total_feature_src, src_temp_feature), 0)
                total_featrue_tar = torch.cat((total_featrue_tar, tar_temp_feature), 0)
                if epoch == 0:
                    total_feature_src0 = torch.cat((total_feature_src0, src_temp_feature), 0)
                    total_featrue_tar0 = torch.cat((total_featrue_tar0, tar_temp_feature), 0)
                    total_classout_tar0 = torch.cat((total_classout_tar0, class_output_tar), 0)
                elif epoch == 1:
                    total_feature_src1 = torch.cat((total_feature_src1, src_temp_feature), 0)
                    total_featrue_tar1 = torch.cat((total_featrue_tar1, tar_temp_feature), 0)
                    total_classout_tar1 = torch.cat((total_classout_tar1, class_output_tar), 0)

                total_domain_pred_src = torch.cat((total_domain_pred_src, domain_pred_src), 0)
                # print('hehhkkdfkdfjdkjfd')
                optimizer.zero_grad()
                # optimizer.zero_grad()
                err.backward()
                # optimizer.step()

                optimizer.step()

                i += 1
            # 保存 epoch0 和epoch1的特征
            if epoch == 0:
                np.save('{}/train_feature0.npy'.format(dic_path),total_feature_src0.cpu().detach())
                np.save('{}/test_feature0.npy'.format(dic_path),
                    total_featrue_tar0.cpu().detach())
                np.save('{}/test_predict_result0.npy'.format(dic_path), total_classout_tar0.cpu().detach())
            elif epoch == 1:
                np.save('{}/train_feature1.npy'.format(dic_path), total_feature_src1.cpu().detach())
                np.save('{}/test_feature1.npy'.format(dic_path),
                        total_featrue_tar1.cpu().detach())
                np.save('{}/test_predict_result1.npy'.format(dic_path), total_classout_tar1.cpu().detach())
            # 贡献因子计算,根据源域的标签聚合索引计算各类别贡献因子均值
            # zero_tensor_src = torch.zeros(total_src_y.size())
            # 统计各种类个数
            contribution_factor = []
            # for fault_type in range(4):
            #     goal_tensor = torch.where( total_src_y == fault_type, total_domain_pred_src, b)
            numpy_total_src_y = total_src_y.cpu().detach().numpy().flatten()
            numpy_total_domain_pred_src = total_domain_pred_src.cpu().detach().numpy()
            # print(type(numpy_total_src_y))
            # print(numpy_total_src_y[:10])
            for fault_type in range(4):
                # contribution_factor.append(numpy_total_domain_pred_src.mean(numpy_total_src_y == fault_type))
                # print('logjia', len(numpy_total_domain_pred_src[np.where(numpy_total_src_y == fault_type)]))
                # print('logyou', np.mean(numpy_total_domain_pred_src))
                contribution_factor.append(
                    np.mean(numpy_total_domain_pred_src[np.where(numpy_total_src_y == fault_type)]))
            # print('logjiayou', contribution_factor)
            np.savetxt(dic_path + '/my_array.csv', contribution_factor, delimiter=',')

            # resutl/ result.csv
            item_pr = 'Epoch: [{}/{}], classify_loss: {:.4f}, domain_loss_s: {:.4f}, domain_loss_t: {:.4f}, ' \
                      'domain_loss: {:.4f},total_loss: {:.4f}, mmd_loss:{:.4f}'.format(
                epoch, args.nepoch, err_s_label.item(), err_s_domain.item(), err_t_domain.item(), err_domain.item(),
                err.item(), mmd_inner.item())
            print(item_pr)
            # print('logxiangchuqu', err_t_domain)
            # print("log77", os.path.exists(os.getcwd() + '/' + args.result_path))
            # fp = open(os.getcwd() + '/' + args.result_path, 'a')
            fp = open( dic_path + args.result_path, 'a')
            fp.write(item_pr + '\n')

            # # test
            acc_src = test(model, 0, epoch, args.source, args.target, args.batchsize)
            acc_tar = test(model, 1, epoch, args.source, args.target, args.batchsize)
            train_acc_list.append(acc_src)
            test_acc_list.append(acc_tar)
            train_loss_list.append(err.item() + mmd_inner)
            mmd_loss_list.append(mmd_inner)
            # acc_src_normal, acc_src_fault1 = test(model, 0, epoch)
            # acc_tar_normal, acc_tar_fault1 = test(model, 1, epoch)
            # test_info = 'Source acc: {:.4f}, target acc: {:.4f}'.format(acc_src, acc_tar)
            # # test_info = 'Source acc_normal: {:.4f}, Source acc_fault1:{:.4f}, target acc_normal: {:.4f}, target acc_fault1: {:.4f}'.format(acc_src_normal, acc_src_fault1,acc_tar_normal,acc_tar_fault1)
            # fp.write(test_info + '\n')
            # print(test_info)
            # fp.close()

            # 绘制伪标签各分类的precision
            one_hot_encoder = preprocessing.OneHotEncoder(categories="auto")
            one_hot_encoder.fit(total_tar_y.cpu().detach().numpy())
            acc_list = metrics.precision_score(total_tar_y.cpu().detach().numpy(), one_hot_encoder.inverse_transform(
                total_classout_tar.cpu().detach().numpy()), average=None)
            # 绘制伪标签各分类的recall
            recall_list = metrics.recall_score(total_tar_y.cpu().detach().numpy(), one_hot_encoder.inverse_transform(
                total_classout_tar.cpu().detach().numpy()), average=None)

            acc_index = 0
            for var in [acc_0, acc_1, acc_2, acc_3]:
                var.append(acc_list[acc_index])
                acc_index += 1
            print('log_acc0', acc_0)

            recall_index = 0
            for recall in [recall_0, recall_1, recall_2, recall_3]:
                recall.append(recall_list[recall_index])
                recall_index += 1

            # # validate
            model.eval()
            validate_alpha = 0
            validate_dataloader = data_loader.load_validate_data(tar=args.target, batch_size=args.batchsize)
            with torch.no_grad():
                for vs_data, vt_data in zip(enumerate(dataloader_src), enumerate(validate_dataloader)):
                    _, (s_f, s_l) = vs_data
                    _, (t_f, t_l) = vt_data
                    s_f, s_l = s_f.to(DEVICE).float(), s_l.to(DEVICE)
                    t_f, t_l = t_f.to(DEVICE).float(), t_l.to(DEVICE)
                    s_l, t_l = s_l.long(), t_l.long()
                    s_l, t_l = s_l.squeeze(1), t_l.squeeze(1)
                    # class_output, err_s_domain, src_temp_feature, domain_pred_src
                    class_out_vs, err_vs_domain, vs_temp_feature, domain_pred_vs = model(input_data=s_f,
                                                                                         alpha=validate_alpha)
                    class_out_vt, err_vt_domain, vt_temp_feature, domain_pred_vt = model(input_data=t_f,
                                                                                         alpha=validate_alpha,
                                                                                         source=False)
                    domian_pred_vt = domain_pred_vt.cpu().detach().numpy()
                    mmd_inner_validate = 0
                    # validate inner mmd start
                    for j in range(4):
                        # 源域特征

                        # 源域特征，非对应标签，替换成零张量
                        temp_f_src = torch.where(s_l == j, vs_temp_feature,
                                                 torch.zeros(vs_temp_feature[0].size()).cuda())
                        temp_index = np.where((torch.max(class_out_vt.data, 1)[1] == j).cpu().detach().numpy())
                        #  计算p(y)
                        # pesudo_prob_validate = len(temp_index[0])/args.batchsize
                        # pesudo_prob_validate = torch.where(torch.max(class_out_vt,1)[1].unsqueeze(1) == j, class_out_vt,
                        #                           torch.zeros(class_out_vt[0].size()).cuda())
                        # pesudo_prob_validate = F.softmax(class_out_vt)
                        pesudo_prob_validate = class_out_vt
                        pesudo_prob_validate = torch.max(pesudo_prob_validate, 1)[0].unsqueeze(1)
                        # print("validate_size", pesudo_prob_validate.size())
                        temp_f_tar = torch.where(torch.max(class_out_vt.data, 1)[1] == j, vt_temp_feature,
                                                 torch.zeros(64, 1).cuda())

                        # 计算样本权重
                        valid_src_factor = torch.where(s_l == j,
                                                       1 / (torch.max(domain_pred_vs) - torch.min(domain_pred_vs))
                                                       * torch.abs(domain_pred_vs - 0.5),
                                                       torch.zeros(domain_pred_vs[0].size()).cuda())

                        # 计算 目标域标签分布概率
                        prob2 = temp_f_tar.size()[0] / args.batchsize

                        # scale_inner = domian_pred_vt / (1 - domian_pred_vt)
                        scale_inner = domian_pred_vt
                        # print("type_scale_inner",type(scale_inner))
                        temp_domain_pred_tar = scale(scale_inner)[temp_index[0]]
                        # temp_domain_pred_tar = scale(scale_inner)[temp_index]
                        # print('ballalal')
                        temp_tar_factor = np.nanmean(temp_domain_pred_tar)

                        if (np.isnan(temp_tar_factor)):
                            print('here nan exist in temp_tar_factor')
                            # temp_tar_factor = 0.5
                            continue

                        # mmd_inner 引入样本权重
                        # single_mmd_validate = mmd_rbf_noaccelerate_ori(temp_f_src,
                        #                                            temp_f_tar, valid_src_factor) * args.mmd_weight
                        # single_mmd_validate = mmd_rbf_noaccelerate_ori(temp_f_src, temp_f_tar) * 1 / real_weight[j] * args.mmd_weight
                        single_mmd_validate = mmd_rbf_noaccelerate_ori(temp_f_src, temp_f_tar, pesudo_prob_validate) * args.mmd_weight
                        mmd_inner_validate = mmd_inner_validate + single_mmd_validate.item()
                    #  validate mmd_inner end
                    domain_pred_vs_mean_abs = torch.abs(torch.mean(domain_pred_vs) - 0.5)
                    # mmd_inner_validate = mmd_rbf_noaccelerate_ori(vs_temp_feature, vt_temp_feature).item()

                    loss_v = loss_class(class_out_vt, t_l)
                    err_domain_validate = err_vs_domain + err_vt_domain
                    err_domain_validate = err_domain_validate * args.gamma
                    valid_losses.append(loss_v.item() + mmd_inner_validate + err_domain_validate.item())
                    # valid_losses.append(loss_v.item() + mmd_inner_validate)
                    # valid_losses.append(
                    #     loss_v.item() + mmd_inner_validate * domain_pred_vs_mean_abs.item() + err_domain_validate.item() * (
                    #                 1 - domain_pred_vs_mean_abs).item())
                valid_loss = np.average(valid_losses)
                valid_loss_epoch.append(valid_loss)
                early_stopping(valid_loss, model)

                # for _, (t_feature, t_label) in enumerate(validate_dataloader):
                #     t_feature, t_label = t_feature.to(DEVICE), t_label.to(DEVICE)
                #     t_feature = t_feature.float()
                #     class_output_v, _, _, _ = model(input_data=t_feature, alpha=alpha)
                #     t_label = t_label.long()
                #     t_label = t_label.squeeze(1)
                #     loss_v = loss_class(class_output_v, t_label)
                #     valid_losses.append(loss_v.item())
                # valid_loss = np.average(valid_losses)
                # early_stopping(valid_loss, model)
        #
        # if best_acc < acc_tar:
        #     best_acc = acc_tar
        #     if not os.path.exists(args.model_path):
        #         os.makedirs(os.getcwd() + '/' + args.model_path)
        #     print('log66此处保存模型')
        #     print("log67", os.path.exists(os.getcwd() + '/' + args.model_path))
        #     # torch.save(model, '{}/mnist_mnistm.pth'.format(args.model_path))
        #     torch.save(model, '{}/model.pth'.format(dic_path))
        #     # # 保存domain_pred
        #     # domain_pred_fatcor = numpy_total_domain_pred_src/(1-numpy_total_domain_pred_src)
        #     # print('domain_pred.size:',domain_pred_fatcor.shape)
        #     # np.save('{}/domain_pred_src.npy'.format(dic_path),domain_pred_fatcor)
        #
        #     # 绘制频率直方图
        #
        #     # 记录遇到最好结果（同一标准，总准确率）时的目标域的各种评价指标
        #     total_tar_y = total_tar_y.cpu().detach()
        #     total_src_y = total_src_y.cpu().detach()
        #     # np.save('/home/zhk/transferLearning/bearing_cppy/bearing/DANN-tr/result/train_label.npy', total_src_y)
        #     # np.save('/home/zhk/transferLearning/bearing_cppy/bearing/DANN-tr/result/test_label.npy', total_tar_y)
        #     np.save('{}/train_label.npy'.format(dic_path), total_src_y)
        #     np.save('{}/test_label.npy'.format(dic_path), total_tar_y)
        #     total_classout_tar = total_classout_tar.cpu().detach()
        #     print('log68', total_tar_y.size())
        #     print('log69', total_classout_tar.size())
        #     one_hot_encoder = preprocessing.OneHotEncoder(categories="auto")
        #     one_hot_encoder.fit(total_tar_y)
        #     test_label_encoder = one_hot_encoder.transform(total_tar_y).toarray()
        #     # one_hot_encoder_train = preprocessing.OneHotEncoder(categories='auto')
        #     # one_hot_encoder_train.fit(total_src_y)
        #     train_label_encoder = one_hot_encoder.transform(total_src_y).toarray()
        #     np.save('{}/train_label_encoder.npy'.format(dic_path), train_label_encoder)
        #     train_predict_result_inverse = one_hot_encoder.inverse_transform(total_classout_src.cpu().detach().numpy())
        #     print('log71', test_label_encoder.shape)
        #     test_predict_result_inverse = one_hot_encoder.inverse_transform(
        #         total_classout_tar)
        #     # 增添 源域预测结果记录
        #     train_y_pred = train_predict_result_inverse
        #     train_y_true = total_src_y.numpy()
        #     test_y_pred = test_predict_result_inverse
        #     test_y_true = total_tar_y.numpy()
        #     container = []
        #     for var in test_y_pred:
        #         container.append(var[0])
        #     print(len(container))
        #     container1 = []
        #     for var in test_y_true:
        #         container1.append(var[0])
        #     print('true', set(container1))
        #     print('pred', set(container))
        #
        #     # dic_path
        #     np.save('{}/train_feature.npy'.format(dic_path),
        #             total_feature_src.cpu().detach())
        #     np.save('{}/test_feature.npy'.format(dic_path),
        #             total_featrue_tar.cpu().detach())
        #     np.save('{}/test_label_encoder.npy'.format(dic_path), test_label_encoder)
        #     np.save('{}/test_predict_result.npy'.format(dic_path), total_classout_tar)
        #     np.save('{}/train_predict_result.npy'.format(dic_path), total_classout_src.cpu().detach().numpy())
        #     np.save('{}/test_predict_result_inverse.npy'.format(dic_path), test_y_pred)
        #     np.save('{}/train_predict_result_inverse.npy'.format(dic_path), train_y_pred)
        #     # path = os.getcwd() + '/result'
        #     path = dic_path
        #     raw_test_y_true = np.load('{}/test_label_encoder.npy'.format(path))
        #     test_y_score = np.load('{}/test_predict_result.npy'.format(path))
        #     raw_train_y_true = np.load('{}/train_label_encoder.npy'.format(path))
        #     train_y_score = np.load('{}/train_predict_result.npy'.format(path))
        #     average_list = ['micro', 'macro', 'weighted', None]
        #     target_names = ['', 'class_0', 'class_1', 'class_2', 'class_3']
        #
        #     auc_score = [metrics.roc_auc_score(raw_test_y_true, test_y_score, average=ways) for ways in average_list]
        #     f1_score = [metrics.f1_score(test_y_true, test_y_pred, average=ways) for ways in average_list]
        #     recall_score = [metrics.recall_score(test_y_true, test_y_pred, average=ways) for ways in average_list]
        #     precision_score = [metrics.precision_score(test_y_true, test_y_pred, average=ways) for ways in average_list]
        #     confusion_matrix = metrics.confusion_matrix(test_y_true, test_y_pred)
        #
        #     # 记录源域结果
        #     auc_score_train = [metrics.roc_auc_score(raw_train_y_true, train_y_score, average=ways) for ways in
        #                        average_list]
        #     f1_score_train = [metrics.f1_score(train_y_true, train_y_pred, average=ways) for ways in average_list]
        #     recall_score_train = [metrics.recall_score(train_y_true, train_y_pred, average=ways) for ways in
        #                           average_list]
        #     precision_score_train = [metrics.precision_score(train_y_true, train_y_pred, average=ways) for ways in
        #                              average_list]
        #     confusion_matrix_train = metrics.confusion_matrix(train_y_true, train_y_pred)
        #     with open(dic_path + '/metrics.txt', 'w') as f:
        #         # auc
        #         f.write('epoch:', epoch, '\n')
        #         f.write('auc : \n')
        #         [f.write('{} : {} \n'.format(average_list[i], auc_score[i])) for i in range(len(average_list))]
        #         f.write('\n')
        #         # f1_score
        #         f.write('f1_score : \n')
        #         [f.write('{} : {} \n'.format(average_list[i], f1_score[i])) for i in range(len(average_list))]
        #         f.write('\n')
        #         # recall_score
        #         f.write('recall_score : \n')
        #         [f.write('{} : {} \n'.format(average_list[i], recall_score[i])) for i in range(len(average_list))]
        #         f.write('\n')
        #         # precision_score
        #         f.write('precision_score : \n')
        #         [f.write('{} : {} \n'.format(average_list[i], precision_score[i])) for i in range(len(average_list))]
        #         f.write('\n')
        #         f.write('confusion_matrix : \n')
        #         for item in target_names:
        #             f.write('{:<10s} '.format(item))
        #         f.write('\n')
        #         for i in range(len(confusion_matrix)):
        #             f.write('{:<10s} '.format(target_names[i + 1]))
        #             for item in confusion_matrix[i]:
        #                 f.write('{:<10d} '.format(item))
        #             f.write('\n')
        #         # for the source
        #         f.write('\n\n')
        #         f.write('---------- for source blew: \n')
        #         [f.write('{} : {} \n'.format(average_list[i], auc_score_train[i])) for i in range(len(average_list))]
        #         f.write('\n')
        #         # f1_score
        #         f.write('f1_score : \n')
        #         [f.write('{} : {} \n'.format(average_list[i], f1_score_train[i])) for i in range(len(average_list))]
        #         f.write('\n')
        #         # recall_score
        #         f.write('recall_score : \n')
        #         [f.write('{} : {} \n'.format(average_list[i], recall_score_train[i])) for i in range(len(average_list))]
        #         f.write('\n')
        #         # precision_score
        #         f.write('precision_score : \n')
        #         [f.write('{} : {} \n'.format(average_list[i], precision_score_train[i])) for i in
        #          range(len(average_list))]
        #         f.write('\n')
        #         f.write('confusion_matrix : \n')
        #         for item in target_names:
        #             f.write('{:<10s} '.format(item))
        #         f.write('\n')
        #         for i in range(len(confusion_matrix_train)):
        #             f.write('{:<10s} '.format(target_names[i + 1]))
        #             for item in confusion_matrix_train[i]:
        #                 f.write('{:<10d} '.format(item))
        #             f.write('\n')
        if early_stopping.early_stop or epoch == args.nepoch - 1:
            print("Early stopping")
            # best_acc = acc_tar
            if not os.path.exists(args.model_path):
                os.makedirs(os.getcwd() + '/' + args.model_path)
            print('log66此处保存模型')
            print("log67", os.path.exists(os.getcwd() + '/' + args.model_path))
            # torch.save(model, '{}/mnist_mnistm.pth'.format(args.model_path))
            # torch.save(model, '{}/model.pth'.format(dic_path))
            torch.save(model.state_dict(), '{}/params_{}.pkl'.format(dic_path, args.source))
            # # 保存domain_pred
            # domain_pred_fatcor = numpy_total_domain_pred_src/(1-numpy_total_domain_pred_src)
            domain_pred_fatcor = numpy_total_domain_pred_src
            print('domain_pred.size:', domain_pred_fatcor.shape)
            np.save('{}/domain_pred_src.npy'.format(dic_path), domain_pred_fatcor)

            # 绘制频率直方图
            index0 = np.where(total_src_y.cpu().detach().numpy() == 0)
            index1 = np.where(total_src_y.cpu().detach().numpy() == 1)
            index2 = np.where(total_src_y.cpu().detach().numpy() == 2)
            index3 = np.where(total_src_y.cpu().detach().numpy() == 3)
            # 0.3,0.75,0.005
            plt.hist(domain_pred_fatcor[index0[0]], bins=np.arange(0.3, 0.75, 0.005), alpha=0.4, label='0')
            plt.hist(domain_pred_fatcor[index1[0]], bins=np.arange(0.3, 0.75, 0.005), alpha=0.4, label='1')
            plt.hist(domain_pred_fatcor[index2[0]], bins=np.arange(0.3, 0.75, 0.005), alpha=0.4, label='2')
            plt.hist(domain_pred_fatcor[index3[0]], bins=np.arange(0.3, 0.75, 0.005), alpha=0.4, label='3')
            plt.title("frequency_histogram")
            plt.legend(loc='upper right')
            plt.savefig('{}/source_frequency.jpg'.format(dic_path))
            plt.clf()

            # 记录遇到最好结果（同一标准，总准确率）时的目标域的各种评价指标
            total_tar_y = total_tar_y.cpu().detach()
            total_src_y = total_src_y.cpu().detach()
            # np.save('/home/zhk/transferLearning/bearing_cppy/bearing/DANN-tr/result/train_label.npy', total_src_y)
            # np.save('/home/zhk/transferLearning/bearing_cppy/bearing/DANN-tr/result/test_label.npy', total_tar_y)
            np.save('{}/train_label.npy'.format(dic_path), total_src_y)
            np.save('{}/test_label.npy'.format(dic_path), total_tar_y)
            total_classout_tar = total_classout_tar.cpu().detach()
            # print('log68', total_tar_y.size())
            # print('log69', total_classout_tar.size())
            one_hot_encoder = preprocessing.OneHotEncoder(categories="auto")
            one_hot_encoder.fit(total_tar_y)
            test_label_encoder = one_hot_encoder.transform(total_tar_y).toarray()
            # one_hot_encoder_train = preprocessing.OneHotEncoder(categories='auto')
            # one_hot_encoder_train.fit(total_src_y)
            train_label_encoder = one_hot_encoder.transform(total_src_y).toarray()
            np.save('{}/train_label_encoder.npy'.format(dic_path), train_label_encoder)
            train_predict_result_inverse = one_hot_encoder.inverse_transform(total_classout_src.cpu().detach().numpy())
            # print('log71', test_label_encoder.shape)
            test_predict_result_inverse = one_hot_encoder.inverse_transform(
                total_classout_tar)
            # 增添 源域预测结果记录
            train_y_pred = train_predict_result_inverse
            train_y_true = total_src_y.numpy()
            test_y_pred = test_predict_result_inverse
            test_y_true = total_tar_y.numpy()
            container = []
            for var in test_y_pred:
                container.append(var[0])
            # print(len(container))
            container1 = []
            for var in test_y_true:
                container1.append(var[0])
            # print('true', set(container1))
            # print('pred', set(container))

            # dic_path
            np.save('{}/train_feature.npy'.format(dic_path),
                    total_feature_src.cpu().detach())
            np.save('{}/test_feature.npy'.format(dic_path),
                    total_featrue_tar.cpu().detach())
            np.save('{}/test_label_encoder.npy'.format(dic_path), test_label_encoder)
            np.save('{}/test_predict_result.npy'.format(dic_path), total_classout_tar)
            np.save('{}/train_predict_result.npy'.format(dic_path), total_classout_src.cpu().detach().numpy())
            np.save('{}/test_predict_result_inverse.npy'.format(dic_path), test_y_pred)
            np.save('{}/train_predict_result_inverse.npy'.format(dic_path), train_y_pred)
            # path = os.getcwd() + '/result'
            path = dic_path
            raw_test_y_true = np.load('{}/test_label_encoder.npy'.format(path))
            test_y_score = np.load('{}/test_predict_result.npy'.format(path))
            raw_train_y_true = np.load('{}/train_label_encoder.npy'.format(path))
            train_y_score = np.load('{}/train_predict_result.npy'.format(path))
            average_list = ['micro', 'macro', 'weighted', None]
            target_names = ['', 'class_0', 'class_1', 'class_2', 'class_3']

            auc_score = [metrics.roc_auc_score(raw_test_y_true, test_y_score, average=ways) for ways in average_list]
            f1_score = [metrics.f1_score(test_y_true, test_y_pred, average=ways) for ways in average_list]
            recall_score = [metrics.recall_score(test_y_true, test_y_pred, average=ways) for ways in average_list]
            precision_score = [metrics.precision_score(test_y_true, test_y_pred, average=ways) for ways in average_list]
            confusion_matrix = metrics.confusion_matrix(test_y_true, test_y_pred)

            # 记录源域结果
            auc_score_train = [metrics.roc_auc_score(raw_train_y_true, train_y_score, average=ways) for ways in
                               average_list]
            f1_score_train = [metrics.f1_score(train_y_true, train_y_pred, average=ways) for ways in average_list]
            recall_score_train = [metrics.recall_score(train_y_true, train_y_pred, average=ways) for ways in
                                  average_list]
            precision_score_train = [metrics.precision_score(train_y_true, train_y_pred, average=ways) for ways in
                                     average_list]
            confusion_matrix_train = metrics.confusion_matrix(train_y_true, train_y_pred)
            with open(dic_path + '/metrics.txt', 'w') as f:
                # auc
                f.write('epoch:{} \n'.format(epoch))
                f.write('auc : \n')
                [f.write('{} : {} \n'.format(average_list[i], auc_score[i])) for i in range(len(average_list))]
                f.write('\n')
                # f1_score
                f.write('f1_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], f1_score[i])) for i in range(len(average_list))]
                f.write('\n')
                # recall_score
                f.write('recall_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], recall_score[i])) for i in range(len(average_list))]
                f.write('\n')
                # precision_score
                f.write('precision_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], precision_score[i])) for i in range(len(average_list))]
                f.write('\n')
                f.write('confusion_matrix : \n')
                for item in target_names:
                    f.write('{:<10s} '.format(item))
                f.write('\n')
                for i in range(len(confusion_matrix)):
                    f.write('{:<10s} '.format(target_names[i + 1]))
                    for item in confusion_matrix[i]:
                        f.write('{:<10d} '.format(item))
                    f.write('\n')
                # for the source
                f.write('\n\n')
                f.write('---------- for source blew: \n')
                [f.write('{} : {} \n'.format(average_list[i], auc_score_train[i])) for i in range(len(average_list))]
                f.write('\n')
                # f1_score
                f.write('f1_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], f1_score_train[i])) for i in range(len(average_list))]
                f.write('\n')
                # recall_score
                f.write('recall_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], recall_score_train[i])) for i in range(len(average_list))]
                f.write('\n')
                # precision_score
                f.write('precision_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], precision_score_train[i])) for i in
                 range(len(average_list))]
                f.write('\n')
                f.write('confusion_matrix : \n')
                for item in target_names:
                    f.write('{:<10s} '.format(item))
                f.write('\n')
                for i in range(len(confusion_matrix_train)):
                    f.write('{:<10s} '.format(target_names[i + 1]))
                    for item in confusion_matrix_train[i]:
                        f.write('{:<10d} '.format(item))
                    f.write('\n')
            plot_lstm_feature.plot(
                dic_path,
                args.source, args.target, True)
            break
    # 绘制伪标签各分类精确率    args.epoch -> 暂停时的epoch
    # plot_for_pesudo(acc_0, acc_1, acc_2, acc_3, dic_path, 'precision',args.nepoch)
    plot_for_pesudo(acc_0, acc_1, acc_2, acc_3, dic_path, 'precison', epoch + 1)
    plot_for_pesudo(recall_0, recall_1, recall_2, recall_3, dic_path, 'recall', epoch + 1)
    #
    print('Test acc: {:.4f}'.format(best_acc))
    # x1 = range(args.nepoch)
    # x2 = range(args.nepoch)
    # x3 = range(args.nepoch)
    x1 = range(epoch + 1)
    x2 = range(epoch + 1)
    x3 = range(epoch + 1)
    y1 = test_acc_list
    y2 = train_loss_list
    # y2_1 = mmd_loss_list
    y3 = train_acc_list
    plt.subplot(4, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(4, 1, 2)
    plt.plot(x2, y2, '.-', label='Training Loss')

    plt.plot(x2, valid_loss_epoch, label='Validate Loss')
    # 绘制checkpoint
    minposs = valid_loss_epoch.index(min(valid_loss_epoch)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.xlabel('total loss vs. epoches')
    plt.ylabel('total loss')
    plt.subplot(4, 1, 3)
    plt.plot(x3, y3, 'o-')
    plt.xlabel('train accuracy vs. epoches')
    plt.ylabel('train accuracy')
    plt.subplot(4, 1, 4)
    plt.plot(x2, mmd_loss_list, '.-', label='mmd_loss')
    plt.xlabel('mmd loss vs. epoches')
    plt.ylabel('mmd loss')
    plt.show()
    plt.savefig("{}/accuracy_loss.jpg".format(dic_path))
    plt.clf()
    print('here')
    # plot_lstm_feature.plot(
    #     dic_path,
    #     args.source, args.target, True)
    print('plot feature')
    # 保存实验参数
    para = {'train': args.source, 'test': args.target, 'lr-center': args.lr_cent, 'lr': args.lr, 'gamma': args.gamma,
            'factor': args.factor, 'batch_size': args.batchsize, 'mmd-weight': args.mmd_weight,
            'patience': args.patience, 'version': args.version,'detail':args.detail}
    with open(dic_path + '/saved_params.json', 'w') as fp:
        json.dump(para, fp)


if __name__ == '__main__':
    torch.random.manual_seed(10)
    # loader_src, loader_tar = data_loader.load_data(src=args.source, tar=args.target, batch_size=args.batchsize)
    loader_src, loader_tar = data_loader_smote.load_data(src=args.source, tar=args.target, batch_size=args.batchsize)
    model = DANN(DEVICE).to(DEVICE)
    if(args.finetue=='true'):
        model.load_state_dict(torch.load('params_{}.pt'.format(args.source)))
    # cent-loss part
    # centerloss = center_loss.CenterLoss(feat_dim=64)
    # params = list(model.parameters()) + list(centerloss.parameters())
    # optimizer = torch.optim.SGD(params, lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)


    def adjust_learning_rate(optimizer_dynamic, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # lr = args.lr * (0.1 ** (epoch // 30))
        lr = args.lr / (1 + 10 * epoch / args.nepoch) ** 0.75
        for param_group in optimizer_dynamic.param_groups:
            param_group['lr'] = lr


    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
    print(len(loader_src), len(loader_tar))
    train(model, optimizer, loader_src, loader_tar, patience=args.patience)
