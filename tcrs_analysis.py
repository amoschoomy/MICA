s="WFGAVILMPYSTNQCKRHDE"
a=[22, 18, 6, 5, 7, 12, 4, 20, 14, 19, 8, 11, 17, 16, 23, 15, 10, 21, 13, 9]
z=["z","z","z","z","z","z","z","z","z","z","z","z","z","z","z","z","z","z","z","z","z","z","z","z"]
print(len(z))
for i in range(20):
    z[a[i]]=s[i]
print(z)
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

a_f=a_f[212:]
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
tcrs=np.array(tcrs).reshape(-1,21)
aa_p=aa_p.reshape(-1,21)
print(tcrs.shape)
print(aa_p.shape)
cancer=[]
for tcr in tcrs:
    c=[]
    for t in tcr:

        c.append(z[t])
    # c=c.replace("z","")
    cancer.append(c)
cancer=np.array(cancer)

cancer_sorted_tcrs=[]
for i in range(20):
   tcrs=[]
   vals, counts=np.unique(cancer[:, i], return_counts=True)
   if vals.shape[0]!=1:
      vals=vals[0:-1]
      counts=counts[0:-1]

   sorted_id=np.argsort(-counts)
   for id in sorted_id:
       tcrs.append(vals[id])
   cancer_sorted_tcrs.append(tcrs)
print(cancer_sorted_tcrs)


commn=[]
for tcr in aa_p:
    c=[]
    for t in tcr:

        c.append(z[t])
    # c=c.replace("z","")
    commn.append(c)
commn=np.array(commn)


commn_sorted_tcrs=[]
for i in range(20):
   tcrs=[]
   vals, counts=np.unique(commn[:, i], return_counts=True)
   if vals.shape[0]!=1:
      vals=vals[0:-1]
      counts=counts[0:-1]

   sorted_id=np.argsort(-counts)
   for id in sorted_id:
       tcrs.append(vals[id])
   commn_sorted_tcrs.append(tcrs)
print(commn_sorted_tcrs)

for i,c in enumerate(cancer_sorted_tcrs):
    print("--------------------"+str(i)+"--------------------")
    print(cancer_sorted_tcrs[i])
    print(commn_sorted_tcrs[i])


