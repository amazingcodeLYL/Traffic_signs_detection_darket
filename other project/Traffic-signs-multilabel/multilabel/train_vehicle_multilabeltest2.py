import os
import re
import shutil
import time
import pickle
import torch
import torchvision
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import multilabel.dataset_read2 as dataset_read
from multilabel.dataset_read2 import color_attrs, direction_attrs, type_attrs
from copy import deepcopy
from PIL import Image
from torchvision.datasets import ImageFolder
from copy import deepcopy
from torchvision import transforms as T
from PIL import Image
from prettytable import PrettyTable
import json


# 此模块负责车辆多标签的训练和测试。训练过程选择交叉熵作为损失函数，需要注意的是，由于是多标签分类
# 故计算loss的时候需要累加各个标签的loss，其中loss = loss_color + loss_direction + 2.0 * loss_type，
# 根据经验，将车辆类型的loss权重放到到2倍效果较好。
# 另一方面，训练分为两步：
# （1）. 冻结除了Resnet-18除全连接层之外的所有层，Fine-tune训练到收敛为止；
# （2）.打开第一步中冻结的所有层，进一步Fine-tune训练，调整所有层的权重，直至整个模型收敛为止。


is_remote = False
use_cuda = True  # True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # users can modify this according to needs and hardware
device = torch.device('cuda:0' if torch.cuda.is_available() and use_cuda else 'cpu')
if use_cuda:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
print('=> device: ', device)

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3)
            Recall = round(TP / (TP + FN), 3)
            Specificity = round(TN / (TN + FP), 3)
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()





class Classifier(torch.nn.Module):
    """
    vehicle multilabel-classifier
    """

    def __init__(self, num_cls, input_size, is_freeze=True):
        """
        :param is_freeze: 冻结层参数
        """
        torch.nn.Module.__init__(self)

        # output channels
        self._num_cls = num_cls

        # input image size
        self.input_size = input_size

        self._is_freeze = is_freeze
        print('=> is freeze: {}'.format(self._is_freeze))

        # delete origin FC and add custom FC
        self.features = torchvision.models.resnet50(pretrained=True)  # True
        del self.features.fc
        # print('feature extractor:\n', self.features)

        self.features = torch.nn.Sequential(
            *list(self.features.children()))

        # self.fc = torch.nn.Linear(512 ** 2, num_cls)  # output channels\
        self.fc = torch.nn.Linear(2048 ** 2, num_cls)


        # print('=> fc layer:\n', self.fc)

        # -----------whether to freeze
        if self._is_freeze:
            for param in self.features.parameters():
                param.requires_grad = False
            # init FC layer
            torch.nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                torch.nn.init.constant_(self.fc.bias.data, val=0)


    def forward(self, X):
        """
        :param X:
        :return:
        """
        N = X.size()[0]

        assert X.size() == (N, 3, self.input_size, self.input_size)

        X = self.features(X)  # extract features

        # print('X.size: ', X.size())
        # assert X.size() == (N, 512, 1, 1)

        X = X.view(N, 2048, 1 ** 2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (1 ** 2)  # Bi-linear CNN for fine-grained classification

        # assert X.size() == (N, 512, 512)

        X = X.view(N, 2048 ** 2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, self._num_cls)
        return X


class Manager(object):
    """
    train and test manager
    """
    def __init__(self, options, path):
        """
        model initialization
        """
        self.options = options
        self.path = path

        # get latest model checkpoint
        if self.options['is_resume']:
            if int(self.path['model_id']) == -1:
                checkpoints = os.listdir(self.path['net'])
                checkpoints.sort(key=lambda x: int(re.match('epoch_(\d+)\.pth', x).group(1)),
                                 reverse=True)
                if len(checkpoints) != 0:
                    self.LATEST_MODEL_ID = int(
                        re.match('epoch_(\d+)\.pth', checkpoints[0]).group(1))
            else:
                self.LATEST_MODEL_ID = int(self.path['model_id'])
        else:
            self.LATEST_MODEL_ID = 0
        print('=> latest net id: {}'.format(self.LATEST_MODEL_ID))

        # net config
        self.net = Classifier(num_cls=51,  # 19 = len(color_attrs) + len(direction_attrs) + len(type_attrs)
                              input_size=224,
                              is_freeze=self.options['is_freeze']).to(device)

        # whether to resume from checkpoint
        if self.options['is_resume']:
            if int(self.path['model_id']) == -1:
                model_path = os.path.join(self.path['net'], checkpoints[0])
            else:
                model_path = self.path['net'] + '/' + \
                    'epoch_' + self.path['model_id'] + '.pth'
            self.net.load_state_dict(torch.load("/home/dell/桌面/PycharmProjects/Traffic-signs-multilabel/multilabel/checkpoints/epoch_108.pth"))
            print('=> net resume from {}'.format(model_path))
        else:
            print('=> net loaded from scratch.')

        # loss function
        self.loss_func = torch.nn.CrossEntropyLoss().to(device)

        # Solver
        if self.options['is_freeze']:
            print('=> fine-tune only the FC layer.')
            self.solver = torch.optim.SGD(self.net.fc.parameters(),
                                          lr=self.options['base_lr'],
                                          momentum=0.9,
                                          weight_decay=self.options['weight_decay'])
        else:
            print('=> fine-tune all layers.')
            self.solver = torch.optim.SGD(self.net.parameters(),
                                          lr=self.options['base_lr'],
                                          momentum=0.9,
                                          weight_decay=self.options['weight_decay'])
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.solver,
        #                                                             mode='max',
        #                                                             factor=0.1,
        #                                                             patience=3,
        #                                                             verbose=True,
        #                                                             threshold=1e-4)

        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.solver,
        #                                                step_size=5,
        #                                                gamma=0.3)


        # train data enhancement
        self.train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                size=self.net.input_size),  # Let smaller edge match
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(
                size=self.net.input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        # test preprocess
        self.test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=self.net.input_size),
            torchvision.transforms.CenterCrop(size=self.net.input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        # load train and test data
        self.train_set = dataset_read.Vehicle(self.path['train_data'],
                                              transform=self.train_transforms,  # train_transforms
                                              train_set="train.txt")
        self.test_set = dataset_read.Vehicle(self.path['test_data'],
                                             transform=self.test_transforms,
                                             train_set="test.txt")

        #Dataloader
        self.train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size=self.options['batch_size'],
                                                        shuffle=True,
                                                        num_workers=4,
                                                        pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=1,  # one image each batch for testing
                                                       shuffle=False,
                                                       num_workers=4,
                                                       pin_memory=True)

        # multilabels
        self.color_attrs=color_attrs
        print('=> color attributes:\n', self.color_attrs)
        self.sign_attrs = direction_attrs
        print('=> signs attributes:\n', self.sign_attrs)
        self.name_attrs=type_attrs
        print('=> name_attributes:\n', self.name_attrs, '\n')

        # for storage and further analysis for err details
        self.err_dict = {}

    def train(self):
        """
        train the network
        """
        print('==> Training...')

        self.net.train()  # train mode
        best_acc = 0.0
        best_epoch = None
        warmup=True
        ricap=True

        epoch_step=[]
        train_acc_=[]
        test_acc_=[]
        print('=> Epoch\tTrain loss\tTrain acc\tTest acc')
        for t in range(self.options['epochs']):  # traverse each epoch
            epoch_step.append(t)

            lr_scheduler=None
            if t == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
                # 学习率预热warmup就是在刚开始训练的时候先使用一个较小的学习率，训练一些epoches或iterations，等模型稳定时再修改为预先设置的学习率进行训练。
                warmup_factor = 1.0 / 1000
                warmup_iters = min(1000, len(self.train_loader) - 1)
                lr_scheduler = self.warmup_lr_scheduler(self.solver, warmup_iters, warmup_factor)
                print(lr_scheduler)

            epoch_loss = []
            num_correct = 0
            num_total = 0

            for data, label  in self.train_loader:  # traverse each batch in the epoch
                print(data,label)

                # put training data, label to device
                data = data.to(device)
                labels = label["labels"].to(device)
                re_label = labels.reshape(-1, 3)
                # # from original paper's appendix
                # if ricap:
                #     I_x, I_y = data.size()[2:]
                #
                #     w = int(np.round(I_x * np.random.beta(0.3, 0.3)))
                #     h = int(np.round(I_y * np.random.beta(0.3, 0.3)))
                #     w_ = [w, I_x - w, w, I_x - w]
                #     h_ = [h, h, I_y - h, I_y - h]
                #
                #     cropped_images = {}
                #     c_ = {}
                #     W_ = {}
                #     for k in range(4):
                #         idx = torch.randperm(data.size(0))
                #         x_k = np.random.randint(0, I_x - w_[k] + 1)
                #         y_k = np.random.randint(0, I_y - h_[k] + 1)
                #         cropped_images[k] = data[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
                #         c_[k] = re_label[idx].cuda()
                #         W_[k] = w_[k] * h_[k] / (I_x * I_y)
                #
                #     patched_images = torch.cat(
                #         (torch.cat((cropped_images[0], cropped_images[1]), 2),
                #          torch.cat((cropped_images[2], cropped_images[3]), 2)),
                #         3)
                #     patched_images = patched_images.cuda()
                    #################


                # clear the grad
                self.solver.zero_grad()
                # forword calculation
                output = self.net.forward(data)

                # statistics of accuracy
                pred= self.get_predict(output)
                labels = labels.cpu().long()
                num_correct += self.count_correct(pred, labels)



                # calculate each attribute loss
                # label = label.long()
                loss_color = self.loss_func(output[:, :4], re_label[:, 0])
                loss_direction = self.loss_func(output[:, 4:8], re_label[:, 1])
                loss_type = self.loss_func(output[:, 8:], re_label[:, 2])
                loss = loss_color + loss_direction + 2.0 * loss_type  # greater weight to type

                # statistics of each epoch loss
                epoch_loss.append(loss.item())

                # statistics of sample number
                num_total += labels.size(0)

                # backward calculation according to loss
                loss.backward()
                self.solver.step()

            # calculate training accuray
            train_acc = 100.0 * float(num_correct) / float(num_total)
            train_acc_.append(train_acc)
            # calculate accuracy of test set
            test_acc = self.test_accuracy(self.test_loader, is_draw=False)
            test_acc_.append(test_acc)

            # schedule the learning rate according to test acc
            # self.scheduler.step(test_acc)
            if lr_scheduler is not None:  # 第一轮使用warmup训练方式
                lr_scheduler.step()  # 使用在optimizer之后调用，不然会改变optimizer中的学习率.step() #使用在optimizer之后调用，不然会改变optimizer中的学习率

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = t + 1

                # dump model to disk
                model_save_name = 'epoch_' + \
                                  str(t + self.LATEST_MODEL_ID + 1) + '.pth'
                print(model_save_name)
                torch.save(self.net.state_dict(),
                           os.path.join(self.path['net'], model_save_name))
                print('<= {} saved.'.format(model_save_name))
            print('\t%d \t%4.3f \t\t%4.2f%% \t\t%4.2f%%' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))

            # statistics of details of each epoch
            # err_dict_path = './err_dict.pkl'
            # pickle.dump(self.err_dict, open(err_dict_path, 'wb'))
            # print('=> err_dict dumped @ %s' % err_dict_path)
            # self.err_dict = {}  # reset err dict

        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.subplot(211)
        plt.plot(epoch_step, train_acc_)

        plt.subplot(212)
        plt.plot(epoch_step, test_acc_)
        plt.savefig("result.png")
        plt.show()
        print('=> Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))

    def test_accuracy(self, data_loader, is_draw=False):
        """
        multi-label test acc
        """

        # 计算混淆矩阵
        # try:
        #     json_file = open('/home/dell/桌面/PycharmProjects/Traffic-signs-multilabel/multilabel/sign.json', 'r')
        #     class_indict = json.load(json_file)
        # except Exception as e:
        #     print(e)
        #     exit(-1)
        # labels = [label for _, label in class_indict.items()]
        # confusion = ConfusionMatrix(num_classes=4, labels=labels)

        self.net.eval()  # test mode

        num_correct = 0
        num_total = 0

        # counters
        num_color = 0
        num_direction = 0
        num_type = 0
        total_time = 0.0

        prohibitory_correct_count=0
        prohibitory_all_count=0

        danger_correct_count=0
        danger_all_count=0

        mandatory_correct_count=0
        mandatory_all_count=0

        other_correct_count=0
        other_all_count=0

        print('=> testing...')
        # for data, label, f_name in data_loader:
        with torch.no_grad():
            for data, label in data_loader:
                # place data in device
                if is_draw:
                    img = data.cpu()[0]
                    img = self.ivt_tensor_img(img)  # Tensor -> image
                # data, label = data.to(device), label.to(device)
                data = data.to(device)
                labels = label["labels"].to(device)

                # format label
                labels = labels.cpu().long()
                start = time.time()
                # forward calculation and processing output
                output = self.net.forward(data)
                pred= self.get_predict(output)  # return to cpu
                # time consuming
                end = time.time()



                total_time += float(end - start)
                if is_draw:
                    print('=> classifying time: {:2.3f} ms'.format(
                        1000.0 * (end - start)))

                # count total number
                num_total += labels.size(0)

                # count each attribute acc
                color_name = self.color_attrs[pred[0][0]]
                sign_name = self.sign_attrs[pred[0][1]]
                # print("sign_name",sign_name)
                # print(labels)
                re_label = labels.reshape(-1)
                # print(re_label)
                if re_label[1]==0:
                    prohibitory_all_count+=1
                elif re_label[1]==1:
                    danger_all_count+=1
                elif re_label[1]==2:
                    mandatory_all_count+=1
                else:
                    other_correct_count+=1

                if pred[0][1]==re_label[1] and re_label[1]==0:
                    prohibitory_correct_count+=1
                elif pred[0][1]==re_label[1] and re_label[1]==1:
                    danger_correct_count+=1
                elif pred[0][1]==re_label[1] and re_label[1]==2:
                    mandatory_correct_count+=1
                else:
                    other_all_count+=1

                # print("labels",re_label[1])

                name_name = self.name_attrs[pred[0][2]]

                if is_draw:
                    fig = plt.figure(figsize=(6, 6))
                    plt.imshow(img)
                    plt.title(color_name + ' ' + sign_name + ' ' + name_name)
                    plt.show()

                num_correct += self.count_correct(pred, labels)
                # num_correct += self.statistics_result(pred, labels)

                # calculate acc of each attribute
                num_color += self.count_attrib_correct(pred, labels, 0)
                num_direction += self.count_attrib_correct(pred, labels, 1)

                num_type += self.count_attrib_correct(pred, labels, 2)
        # confusion.plot()
        # confusion.summary()
        # calculate time consuming of inference
        print('=> average inference time: {:2.3f} ms'.format(
            1000.0 * total_time / float(len(data_loader))))

        accuracy = 100.0 * float(num_correct) / float(num_total)
        color_acc = 100.0 * float(num_color) / float(num_total)
        direction_acc = 100.0 * float(num_direction) / float(num_total)
        type_acc = 100.0 * float(num_type) / float(num_total)

        prohibitory_acc=100.0*float(prohibitory_correct_count)/float(prohibitory_all_count)
        danger_acc=100.0*float(danger_correct_count)/float(danger_all_count)
        mandatory_acc=100.0*float(mandatory_correct_count)/float(mandatory_all_count)
        other_acc=100.0*float(other_correct_count)/float(other_all_count)

        print('=> prohibitory_acc: {:.3f}% , danger_acc: {:.3f}% ,mandatory_acc: {:.3f}% ,other_acc: {:.3f}% '.format(prohibitory_acc,danger_acc,mandatory_acc,other_acc))
        print('=> test accuracy: {:.3f}% | color acc: {:.3f}%, signs acc: {:.3f}%, classes acc: {:.3f}%'.format(
            accuracy, color_acc, direction_acc, type_acc))
        return accuracy

    def get_predict(self, output):
        """
        processing output
        :param output:
        :return: prediction
        """
        # get prediction for each label
        output = output.cpu()  # get data back to cpu side
        pred_color = output[:, :4]
        pred_direction = output[:, 4:8]
        pred_type = output[:, 8:]

        color_idx = pred_color.max(1, keepdim=True)[1]
        direction_idx = pred_direction.max(1, keepdim=True)[1]
        type_idx = pred_type.max(1, keepdim=True)[1]
        pred = torch.cat((color_idx, direction_idx, type_idx), dim=1)
        return pred

    def count_correct(self, pred, label):
        """
        :param pred:
        :param label:
        :return:
        """
        label_cpu = label.cpu().long()  # 需要将label转化成long tensor
        assert pred.size(0) == label.size(0)
        correct_num = 0
        for one, two in zip(pred, label):
            two = two.reshape(-1)
            if torch.equal(one, two):
                correct_num += 1
        return correct_num

    def statistics_result(self, pred, label):
        """
        statistics of correct and error
        :param pred:
        :param label:
        :param f_name:
        :return:
        """
        # label_cpu = label.cpu().long()
        assert pred.size(0) == label.size(0)
        correct_num = 0
        err_result = {}
        for one, two in zip(pred, label):
            if torch.equal(one, two):  # statistics of correct number
                correct_num += 1
            else:  # statistics of detailed error info
                pred_color = self.color_attrs[one[0]]
                pred_sign = self.sign_attrs[one[1]]
                pred_name = self.name_attrs[one[2]]

                two = two.reshape(-1)
                label_color = self.color_attrs[two[0]]
                label_sign = self.sign_attrs[two[1]]
                label_name = self.name_attrs[two[2]]

                err_result = label_color + ' ' + label_sign + ' ' + label_name + \
                             ' => ' + \
                             pred_color + ' ' + pred_sign + ' ' + pred_name
                # self.err_dict[name] = err_result
        return correct_num

    def warmup_lr_scheduler(self,optimizer, warmup_iters, warmup_factor):

        def f(x):
            """根据step数返回一个学习率倍率因子"""
            if x >= warmup_iters:  # 当迭代数大于给定的warmup_iters时，倍率因子为1
                return 1
            alpha = float(x) / warmup_iters
            # 迭代过程中倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

    def count_attrib_correct(self, pred, label, idx):
        """
        :param pred:
        :param label:
        :param idx:
        :return:
        """
        assert pred.size(0) == label.size(0)
        correct_num = 0
        for one, two in zip(pred, label):
            # print(one,two)
            two = two.reshape(-1)
            if one[idx] == two[idx]:
                correct_num += 1
        return correct_num

    def ivt_tensor_img(self, inp, title=None):
        """
        Imshow for Tensor.
        """

        # turn channelsxWxH into WxHxchannels
        inp = inp.numpy().transpose((1, 2, 0))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # de-standardization
        inp = std * inp + mean

        # clipping
        inp = np.clip(inp, 0, 1)

        # plt.imshow(inp)
        # if title is not None:
        #     plt.title(title)
        # plt.pause(0.001)  # pause a bit so that plots are updated
        return inp

    def recognize_pil(self, image):
        """
        classify a single image
        :param img: PIL Image
        :return:
        """
        img = deepcopy(image)
        if img.mode == 'L' or img.mode == 'I':  # turn 8bits or 32bits gray into RGB
            img = img.convert('RGB')
        img = self.test_transforms(img)
        img = img.view(1, 3, self.net.module.input_size,
                       self.net.module.input_size)

        # put data to device
        img = img.to(device)

        start = time.time()

        # inference calculation
        output = self.net.forward(img)

        # get prediction
        pred = self.get_predict(output)

        end = time.time()

        print('=> classifying time: {:2.3f} ms'.format(1000.0 * (end - start)))

        color_name = self.color_attrs[pred[0][0]]
        sign_name = self.sign_attrs[pred[0][1]]
        name_name = self.name_attrs[pred[0][2]]

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(color_name + ' ' + sign_name + ' ' + name_name)
        plt.show()

    def test_single(self):
        """
        test single image
        :return:
        """
        self.net.eval()

        root = './test_imgs/test_GTSRB'
        for file in os.listdir(root):
            file_path = os.path.join(root, file)
            image = Image.open(file_path)
            self.recognize_pil(image)

    def random_pick(self, src, dst, pick_num=20):
        """
        random pick from src to dst
        :param src:
        :param dst:
        :return:
        """
        if not os.path.exists(src) or not os.path.exists(dst):
            print('=> [Err]: invalid dir.')
            return

        if len(os.listdir(dst)) != 0:
            shutil.rmtree(dst)
            os.mkdir(dst)

        # recursive traversing, search for '.jpg'
        jpgs_path = []

        def find_jpgs(root, jpgs_path):
            """
            :param root:
            :param jpgs_path:
            :return:
            """
            for file in os.listdir(root):
                file_path = os.path.join(root, file)

                if os.path.isdir(file_path):  # if dir do recursion
                    find_jpgs(file_path, jpgs_path)
                else:  # if file, put to list
                    if os.path.isfile(file_path) and file_path.endswith('.jpg'):
                        jpgs_path.append(file_path)

        find_jpgs(src, jpgs_path)
        # print('=> all jpgs path:\n', jpgs_path)

        # no replace random pick
        pick_ids = np.random.choice(
            len(jpgs_path), size=pick_num, replace=False)
        for id in pick_ids:
            shutil.copy(jpgs_path[id], dst)


def run():
    """
    main loop function
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='Train bi-linear CNN based vehicle multilabel classification.')
    parser.add_argument('--base_lr',
                        dest='base_lr',
                        type=float,
                        default=1.0,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size',
                        dest='batch_size',
                        type=int,
                        default=32,  # resnet18 256  resnet50 64
                        help='Batch size.')  # 用多卡可以设置的更大
    parser.add_argument('--epochs',
                        dest='epochs',
                        type=int,
                        default=100,
                        help='Epochs for training.')
    parser.add_argument('--weight_decay',
                        dest='weight_decay',
                        type=float,
                        default=0.0005,
                        help='Weight decay.')
    parser.add_argument('--use-cuda', type=bool, default=True,
                        help='whether to use GPU or not.')
    parser.add_argument('--is-freeze',
                        type=bool,
                        default=True,
                        help='whether to freeze all other layers except FC layer.')
    parser.add_argument('--is-resume',
                        type=bool,
                        default=False,
                        help='whether to resume from checkpoints')
    parser.add_argument('--pre-train',
                        type=bool,
                        default=True,
                        help='whether in pre training mode.')
    args = parser.parse_args()

    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must > 0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must > 0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must > 0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must > 0.')

    if args.pre_train:
        options = {
            'base_lr': args.base_lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'weight_decay': args.weight_decay,
            'is_freeze': False,
            'is_resume': False
        }
    else:
        options = {
            'base_lr': args.base_lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'weight_decay': args.weight_decay,
            'is_freeze': False,
            'is_resume': False
        }

    # super parameters for fine-tuning
    if not options['is_freeze']:
        options['base_lr'] = 1.0
        options['epochs'] = 110
        options['weight_decay'] = 1e-8  # 1e-8
    print('=> options:\n', options)

    parent_dir = os.path.realpath(
        os.path.join(os.getcwd(), '..')) + os.path.sep
    project_root = parent_dir
    print('=> project_root: ', project_root)

    path = {
        'net': '/home/dell/桌面/PycharmProjects/Traffic-signs-multilabel/multilabel/checkpoints/',
        'model_id': '-1',
        'train_data': '/home/dell/桌面',
        'test_data': '/home/dell/桌面'
    }

    manager = Manager(options, path)
    manager.train()
    # manager.test_accuracy(manager.test_loader, is_draw=True)
    # manager.random_pick(src='./test_imgs/test_GTSRB', dst='./test_result')
    # manager.test_single()






if __name__ == '__main__':
    run()

