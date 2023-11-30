import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
# 假设 intermediate_output 是从模型中获取的中间层输出
# 这里使用随机生成的数据作为示例
a_f = np.load("a_f.npy")  # 100个样本，50个特征
# a_f=a_f[:,55:65]
a_y=np.load("a_y.npy")
a_y=torch.tensor(a_y)
a_y,i=torch.sort(a_y)
one=a_y[a_y==1]

aa_f=[]
iiii=i
for ii in i:

    aa_f.append(a_f[ii])
a_f= np.array(aa_f)  # 100个样本，50个特征


a_w=np.load("a_w.npy")




a_b=np.load("a_b.npy")

a_f=a_f[212:-1]
# a_f=a_f[-1]
a_l=[]
for i_f in a_f:

    l = []
    for i, f in enumerate(i_f):
        d = []
        no = f * a_w[0][i]
        yes = f * a_w[1][i]
        d.append(no)
        d.append(yes)
        l.append(d)
    a_l.append(l)
a_l=np.array(a_l)

a_l=np.sum(a_l,axis=0)

# for i,f in enumerate(a_f):
#     d=[]
#     no=f*a_w[0][i]
#     yes=f*a_w[1][i]
#     d.append(no)
#     d.append(yes)
#     l.append(d)
# l=np.array(l)
# l=l[:,1]

sns.heatmap(a_l,   cmap='coolwarm', fmt='.2f', linewidths=.5 )
#sns.heatmap(intermediate_output, cmap='viridis', cbar=True)
#sns.heatmap(data, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
#plt.title('Feature heatmap of unsort')
plt.xlabel('Classification')
plt.ylabel('Instance Index')
plt.savefig("classification_heatmap.jpg")
plt.show()
var=0
for l in a_l:

    var=var+np.std(l)
var =var/100
tcr_index=[]
for i,l in enumerate(a_l):

    if np.std(l)>=var:
        tcr_index.append(i)

a_d=np.load("a_d.npy")

tcr_or_index=[]
for i_d in a_d:
    or_index=[]
    for index in tcr_index:
        or_index.append(i_d[index])
    tcr_or_index.append(or_index)
tcr_or_index=np.array(tcr_or_index)

a_p=np.load("a_p.npy")

aa_p=[]
for ii in iiii:

    aa_p.append(a_p[ii])
a_p= np.array(aa_p)  # 100个样本，50个特征
aa_p=a_p

a_p=a_p[212:-1]
aa_p=aa_p[0:212]

tcrs=[]
for i,p in enumerate(a_p):
    t_p=[]
    for index in tcr_or_index[i]:
         t_p.append(p[index])
    tcrs.append(t_p)
tcrs=np.array(tcrs)

most=[]
all_cancer=[]
cancer_a_p=np.zeros([24,])

for i in range(21):

    vals, counts = np.unique(tcrs[:,:,i], return_counts=True)

    for c_i,c in enumerate(counts):
        cancer_a_p[vals[c_i]]= cancer_a_p[vals[c_i]]+c/(tcrs.shape[0]*tcrs.shape[1]*tcrs.shape[2])
    all_cancer.append(counts/np.sum(counts))
    index = np.argmax(counts)  # 获得出现次数最多的索引
    most.append(vals[index])
    #  second_most_freq_index = np.argsort(-counts)[1]  # 第二多的值在排序后的索引
    #  second_most_freq_value = vals[second_most_freq_index]
    #  if second_most_freq_value !=2:
    #   second_most_freq_index = np.argsort(-counts)[1]  # 第二多的值在排序后的索引
    #   second_most_freq_value = vals[second_most_freq_index]
    #   most.append(second_most_freq_value)
# print(most)

most=[]
all_common=[]
common_a_p=np.zeros([24,])
for i in range(21):


    vals, counts = np.unique(aa_p[:,:,i], return_counts=True)
    for c_i,c in enumerate(counts):
        common_a_p[vals[c_i]]= common_a_p[vals[c_i]]+c/(aa_p.shape[0]*aa_p.shape[1]*aa_p.shape[2])

    index = np.argmax(counts)  # 获得出现次数最多的索引
    most.append(vals[index])
    # all_cancer.append(counts/np.sum(counts))
    # index = np.argmax(counts)  # 获得出现次数最多的索引
    #  all_common.append(np.argsort(-counts))
    #  second_most_freq_index = np.argsort(-counts)[1]  # 第二多的值在排序后的索引
    #  second_most_freq_value = vals[second_most_freq_index]
    #  if second_most_freq_value !=2:
    #   second_most_freq_index = np.argsort(-counts)[1]  # 第二多的值在排序后的索引
    #   second_most_freq_value = vals[second_most_freq_index]
    #   most.append(second_most_freq_value)
# print(most)
# print(np.array(all_common))
print(cancer_a_p)
print(common_a_p)
cancer_a_p=torch.tensor(cancer_a_p)
common_a_p=torch.tensor(common_a_p)
cancer_a_p,cancer_index=torch.sort(cancer_a_p)
print(cancer_index)
print(cancer_a_p[5])
common_a_p,common_index=torch.sort(common_a_p)
print(common_index)
print(common_a_p[5])


