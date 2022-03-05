import os
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.model_selection import KFold
# from tensorboardX import SummaryWriter
from models.CNN_RNN import Classifier_Net
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from models.Small_TCN import Small_TCN

# hyper parameters
GPU_NUM = 0
EPOCHS = 100
BATCH_SIZE = 16
Learning_rate = 0.00005


def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
    # if isinstance(m, nn.Linear):
    #     m.weight.data.normal_(0, 0.01)
    #     m.bias.data.zero_()


def ECGplot(data):
    fig = plt.figure(figsize=(8, 4))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax4 = plt.subplot2grid((2, 2), (1, 1))

    ax1.plot(data[0], linewidth=0.7, label='type (7)')
    ax1.legend(loc='upper right',
               fontsize='small')  # fontsize : int or float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    # ax1.set_ylabel('Amplitude')
    ax2.plot(data[1], linewidth=0.7, label='HMCKR with binary tree encoding')
    ax2.legend(loc='lower right', fontsize='small')
    # ax2.set_ylabel('Amplitude')

    ax3.plot(data[2], linewidth=0.7, label='type (4)')
    ax3.legend(loc='upper right', fontsize='small')

    ax4.plot(data[3], linewidth=0.7, label='HMCKR with Huffman encoding')
    ax4.legend(loc='lower right', fontsize='small')

    plt.show()

    return


def cal_results(Label_all, predict_y_all):
    results = []
    acc_macro = accuracy_score(Label_all, predict_y_all)
    pre_macro = precision_score(Label_all, predict_y_all, average='macro')
    rec_macro = recall_score(Label_all, predict_y_all, average='macro')
    f1_macro = f1_score(Label_all, predict_y_all, average='macro')
    results.append(acc_macro)
    results.append(pre_macro)
    results.append(rec_macro)
    results.append(f1_macro)

    acc_micro = accuracy_score(Label_all, predict_y_all)
    pre_micro = precision_score(Label_all, predict_y_all, average='micro')
    rec_micro = recall_score(Label_all, predict_y_all, average='micro')
    f1_micro = f1_score(Label_all, predict_y_all, average='micro')
    results.append(acc_micro)
    results.append(pre_micro)
    results.append(rec_micro)
    results.append(f1_micro)

    acc_wei = accuracy_score(Label_all, predict_y_all)
    pre_wei = precision_score(Label_all, predict_y_all, average='weighted')
    rec_wei = recall_score(Label_all, predict_y_all, average='weighted')
    f1_wei = f1_score(Label_all, predict_y_all, average='weighted')
    results.append(acc_wei)
    results.append(pre_wei)
    results.append(rec_wei)
    results.append(f1_wei)

    # print(confusion_matrix(Label_all, predict_y_all))

    matrix = confusion_matrix(Label_all, predict_y_all)

    return results, matrix

def train_net_classifier(save_path, MAX_ACC):
    # ==read data==
    # label = scio.loadmat('data/Train_Test/Label_data')
    # label = label['data']
    label = np.load('data/ECG_train_label.npy')
    # label = np.load('data/ECG_train_label_21445.npy')
    # label = np.load('data/Label.npy')
    # mat = np.load('data/ECG_train_data_nom.npy')
    mat = np.load('data/ECG_train_data.npy')
    # mat = np.load('data/ECG_train_data_21445.npy')
    # mat = np.load('data/Data.npy')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # store = pd.HDFStore('data/Train_Test/ECG_data_float32_z_score_l.h5')
    # df1 = store['data']
    # mat = np.array(df1)
    print(mat.shape)
    # store.close()

    kf = KFold(n_splits=10, shuffle=True, random_state=1)

    # mat = np.concatenate((mat, label[:,0:1]), axis=1)
    # print(mat.shape)

    fold = 0

    for train_index, test_index in kf.split(mat):
        fold = fold + 1
        # print('train_index', train_index, 'test_index', test_index)
        # print('train_index', 'test_index')
        train_data, train_y = mat[train_index], label[train_index]
        test_data, test_y = mat[test_index], label[test_index]

        # train_X = mat[train_index]
        # test_X = mat[test_index]
        # train_X = train_X[:,:360000]
        # test_X = test_X[:,:360000]
        # train_y = train_X[:,360000:]
        # test_y = test_X[:,360000:]

        # train_data = np.reshape(train_X, (len(train_X), 12, 3600))
        # #(12,2500,12)
        # # train_data = train_data[:, 1, 2:4, :]
        # test_data = np.reshape(test_X, (len(test_X), 12, 3600))
        # # test_data = test_data[:, 1, 2:4, :]
        # train_y = train_y[:, 0].squeeze()  # use CrossEntropy, label must be 1 dim
        # test_y = test_y[:, 0].squeeze()  # use CrossEntropy, label must be 1 dim

        print(train_data.shape, train_y.shape, test_data.shape, test_y.shape)

        Traindata = torch.from_numpy(train_data).float()  # transform to float torchTensor   train_data[47000:]
        Testdata = torch.from_numpy(test_data).float()
        Traindata_Label = torch.from_numpy(train_y).long()  # transform to float torchTensor     train_label[47000:]
        Testdata_L = torch.from_numpy(test_y).long()  # transform to float torchTensor

        # Dataset
        # TorchDataset = Data.TensorDataset(data_tensor=Traindata, target_tensor=Traindata_Label)
        TorchDataset = Data.TensorDataset(Traindata, Traindata_Label)
        TorchDataset_Test = Data.TensorDataset(Testdata, Testdata_L)

        test_num = len(TorchDataset_Test)

        # Data Loader for easy mini-batch return in training
        Train_loader = Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)
        Test_loader = Data.DataLoader(dataset=TorchDataset_Test, batch_size=BATCH_SIZE, shuffle=True)

        # CNet = Classifier_Net(num_classes=9)
        # CNet = TCN(12, 9, [25] * 12, kernel_size=32, dropout=0.05)
        # CNet.load_state_dict(torch.load(''))
        # CNet.apply(weigth_init)
        CNet = Small_TCN().to(device=device)
        print(CNet)

        optimizer_net = torch.optim.Adam(CNet.parameters(), lr=Learning_rate)

        loss_MSE = nn.MSELoss()
        loss_PairD = nn.PairwiseDistance()
        loss_CrossEn = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            running_loss = 0  # observe loss

            for step, (X, Y) in enumerate(Train_loader):
                # print(X,'  nnnn   ',Y)

                X = Variable(X).to(device=device)
                Y = Variable(Y).to(device=device)

                C_gen = CNet(X)

                # print('C_gen : ',C_gen)
                # loss_CrossEn = nn.CrossEntropyLoss()

                loss_cl = loss_CrossEn(C_gen, Y)
                print(C_gen[0], loss_cl, step)
                running_loss += loss_cl.item()

                # print(running_loss, step)
                # exit(0)

                optimizer_net.zero_grad()
                loss_cl.backward()
                optimizer_net.step()

            rate = (epoch + 1) / EPOCHS
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)

            print(
                "\rtrain loss: {:^4.0f}%[{}->{}]  {:.6f}".format(int(rate * 100), a, b, running_loss / step, end="\n"))
            # print(epoch,'Epoch finished ! Loss: {:.6f}'.format(running_loss / step))
            # writer.add_scalar('Loss/loss', running_loss / step, epoch)

            # validate
            CNet.eval()
            CNet.eval()

            acc = 0.0  # accumulate accurate number / epoch
            predict_y_all = []
            # y_scores = []
            Label_all = []

            with torch.no_grad():
                for val_data in Test_loader:
                    val_data, val_labels = val_data
                    outputs = CNet(val_data.to(device=device))  # eval models only have last output layer
                    # loss = loss_function(outputs, test_labels)
                    # predict_y = torch.max(outputs.cpu(), dim=1)[1]
                    predict_y = torch.max(outputs.cpu(), 1)[1].data.numpy() #
                    # val_C_label = val_labels[:, (tree_deep * dim):]  # .contiguous()
                    # val_C_label = val_C_label.squeeze().long()
                    acc += sum(1 for a, b in zip(predict_y, val_labels.to(device=device)) if a == b) #.to(device=device)
                    # acc += sum(predict_y == val_labels.cuda(GPU_NUM))

                    predict_y_all = np.concatenate((predict_y_all, predict_y), axis=0)

                    Label_all = np.concatenate((Label_all, val_labels.cpu().data.numpy()), axis=0)#.to(device=device)

                val_accurate = acc / test_num

                results, matrix = cal_results(Label_all, predict_y_all)

                print(val_accurate, results, '\n', matrix)

                if val_accurate > MAX_ACC:
                    MAX_ACC = val_accurate
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(CNet.state_dict(), save_path + 'best_model_fold{}.pth'.format(fold))
                    f = open(r'matrix.txt' , 'w')
                    for i in range(len(matrix)):
                        for j in range(len(matrix)):
                            f.write(str(matrix[i][j])+'\t')
                        f.write('\n')
                    f.close()
                    print("Net saved !!!")
                print('[epoch %d] train_loss: %.5f  test_accuracy==================================>: %.5f'
                      % (epoch + 1, running_loss / step, val_accurate))

            CNet.train()
            CNet.train()
        break
    print('Finished Training')


if __name__ == '__main__':
    save_path = '../ckpts_classifier/'

    max_ACC = 0.87

    train_net_classifier(save_path, max_ACC)
