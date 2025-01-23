import numpy as np
import torch
from sklearn import metrics

from torch import nn
from torch.utils.data import DataLoader, Dataset

train_data = np.load("/scratch/project/tcr_ml/MICA/data/data/Lung/lung_train_data.npy")
train_label = np.load("/scratch/project/tcr_ml/MICA/data/data/Lung/lung_train_label.npy")
test_data = np.load("/scratch/project/tcr_ml/MICA/data/data/Lung/lung_test_data.npy")
test_label = np.load("/scratch/project/tcr_ml/MICA/data/data/Lung/lung_test_label.npy")
all_data = []
all_label = []
for p in train_data:
    all_data.append(p)
for l in train_label:
    all_label.append(l)
for p in test_data:
    all_data.append(p)
for l in test_label:
    all_label.append(l)
all_data = np.array(all_data)
all_label = np.array(all_label)
feature_number = all_data.shape[0]
print(feature_number)
vn_idx = range(0, feature_number)

nn_s = int(np.ceil(feature_number * (1 - 0.2)))


class Mydata(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        gene = torch.from_numpy(self.data[index])

        return gene.to(torch.int), int(self.label[index])

    def __len__(self):
        return self.data.shape[0]


vn_train = np.random.choice(vn_idx, nn_s, replace=False)

vn_test = [x for x in vn_idx if x not in vn_train]
train_data = np.array(all_data)[vn_train]
test_data = np.array(all_data)[vn_test]
train_label = np.array(all_label)[vn_train]
test_label = np.array(all_label)[vn_test]
print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)
my_train = Mydata(train_data, train_label)
my_test = Mydata(test_data, test_label)
dataloader_train = DataLoader(dataset=my_train, batch_size=356, shuffle=True)
dataloader_test = DataLoader(dataset=my_test, batch_size=88, shuffle=True)


def caculateAUC(AUC_outs, AUC_labels):
    ROC = 0
    outs = []
    labels = []
    for (index, AUC_out) in enumerate(AUC_outs):
        softmax = nn.Softmax(dim=1)
        out = softmax(AUC_out).detach().numpy()
        out = out[:, 1]
        for out_one in out.tolist():
            outs.append(out_one)
        for AUC_one in AUC_labels[index].tolist():
            labels.append(AUC_one)

    outs = np.array(outs)

    labels = np.array(labels)
    fpr, tpr, thresholds = metrics.roc_curve(labels, outs, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(labels, outs)

    return auc, aupr


#python3  train_and_test.py
class MICA(nn.Module):

    def __init__(self, ):
        super(MICA, self).__init__()

        self.embedding = nn.Embedding(40, 50)  # .frm_pretrained(torch.load("embedding_token"))

        self.conv1 = nn.Conv2d(1, 3, kernel_size=(5, 50), stride=1)
        nn.init.xavier_normal_(self.conv1.weight, gain=1)

        self.conv2 = nn.Conv2d(1, 3, kernel_size=(3, 50), stride=1)
        nn.init.xavier_normal_(self.conv2.weight, gain=1)

        self.conv3 = nn.Conv1d(1, 1, kernel_size=(3,), stride=1)
        nn.init.xavier_normal_(self.conv3.weight, gain=1)
        self.conv4 = nn.Conv1d(1, 1, kernel_size=(5,), stride=1)
        nn.init.xavier_normal_(self.conv4.weight, gain=1)

        self.conv5 = nn.Conv1d(1, 1, kernel_size=(7,), stride=1)
        nn.init.xavier_normal_(self.conv5.weight, gain=1)

        self.conv6 = nn.Conv2d(1, 3, kernel_size=(7, 50), stride=1)
        nn.init.xavier_normal_(self.conv6.weight, gain=1)

        self.Flatten = nn.Flatten()

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(153, 1)
        nn.init.xavier_normal_(self.linear1.weight, gain=1)

        self.linear2 = nn.Linear(100, 2)

        # self.linear3=nn.Linear(100,2)
        self.BatchNorm1 = nn.LazyBatchNorm2d()

        self.BatchNorm2 = nn.LazyBatchNorm2d()
        self.BatchNorm3 = nn.LazyBatchNorm1d()
        self.BatchNorm4 = nn.LazyBatchNorm1d()
        self.BatchNorm5 = nn.LazyBatchNorm1d()
        self.BatchNorm6 = nn.LazyBatchNorm2d()
        self.W_q = nn.Linear(1, 1)
        self.W_k = nn.Linear(1, 1)
        self.W_v = nn.Linear(1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.75)
        self.max = nn.MaxPool2d(kernel_size=(2, 1), stride=2)

    def forward(self, input):
        ap = input
        input = self.embedding(input)
        input = input.reshape(input.shape[0] * input.shape[1], 1, input.shape[2], input.shape[3])

        conv1 = self.conv1(input)
        conv1 = self.relu(conv1)
        conv1 = self.BatchNorm1(conv1)
        conv1 = self.dropout(conv1)
        # conv1 = self.max(conv1)
        conv1 = self.Flatten(conv1)

        conv2 = self.conv2(input)
        conv2 = self.relu(conv2)
        conv2 = self.BatchNorm2(conv2)
        conv2 = self.dropout(conv2)
        # conv2 = self.max(conv2)
        conv2 = self.Flatten(conv2)

        conv6 = self.conv6(input)
        conv6 = self.relu(conv6)
        conv6 = self.BatchNorm6(conv6)
        conv6 = self.dropout(conv6)
        # conv2 = self.max(conv2)
        conv6 = self.Flatten(conv6)

        all = torch.cat([conv2, conv1, conv6], 1)

        all1 = self.linear1(all)
        all1 = all1.reshape(-1, 100)
        all1, ad = torch.sort(all1)
        a2 = all1.reshape(-1, 100)
        all1 = all1.reshape(-1, 100, 1)

        q = self.W_q(all1)
        k = self.W_k(all1)
        v = self.W_v(all1)


        attention_scores = torch.matmul(q, k.transpose(1, 2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(100))



        attention_scores = self.softmax(attention_scores)


        all1 = torch.matmul(attention_scores, v)
        a = all1.reshape(-1, 100).cpu().detach().numpy()
        all1 = self.linear2(all1.reshape(-1, 100))
        weights = self.linear2.weight.data
        biases = self.linear2.bias.data
        return all1, a, a2, weights, biases, ap, ad


def train_one_epoch(model, dataloader_train, optimizer, loss_fn, epoch, my_train):
    train_acc = 0.0
    model.train()
    for data in dataloader_train:
        input, label = data
        input = input.cuda()
        label = label.cuda()
        
        out, a_f, a_i, w, b, ap, ad = model(input)
        if (epoch + 1) % 100 == 0:
            np.save("a_f", a_f)
            np.save("a_y", label.cpu().detach().numpy())
            np.save("a_i", a_i.cpu().detach().numpy())
            np.save("a_w", w.cpu().detach().numpy())
            np.save("a_b", b.cpu().detach().numpy())
            np.save("a_p", ap.cpu().detach().numpy())
            np.save("a_d", ad.cpu().detach().numpy())
        
        optimizer.zero_grad()
        result_loss = loss_fn(out, label)
        result_loss.backward()
        optimizer.step()
        train_acc += (out.argmax(1) == label).sum()
    
    return float(train_acc / my_train.__len__())

def evaluate(model, dataloader_test, loss_fn, my_test):
    model.eval()
    auc_label = []
    auc_out = []
    test_acc = 0.0
    
    with torch.no_grad():
        for data in dataloader_test:
            input, label = data
            input = input.cuda()
            label = label.cuda()
            out, _, t_a_i, tw, tb, tp, td = model(input)
            
            result_loss = loss_fn(out, label)
            
            auc_label.append(label.cpu().numpy())
            auc_out.append(out.cpu())
            test_acc += (out.argmax(1) == label).sum()
    
    auc_number, aupr = caculateAUC(auc_out, auc_label)
    return float(test_acc / my_test.__len__()), auc_number

# Model initialization and training setup
model = MICA()
model = model.cuda()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001, weight_decay=5e-3)
train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[14000], gamma=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

# Main training loop
for epoch in range(14000):
    train_acc = train_one_epoch(model, dataloader_train, optimizer, loss_fn, epoch, my_train)
    train_scheduler.step()
    
    if (epoch + 1) % 500 == 0:
        print(f"Training accuracy: {train_acc}")
        test_acc, auc_number = evaluate(model, dataloader_test, loss_fn, my_test)
        print(f"Test accuracy: {test_acc}, AUC: {auc_number}")
