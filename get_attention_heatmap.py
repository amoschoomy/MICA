import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
# 假设 intermediate_output 是从模型中获取的中间层输出
# 这里使用随机生成的数据作为示例
a_f = np.load("a_f.npy")  # 100个样本，50个特征
a_f=a_f[:,28:78]
a_y=np.load("a_y.npy")
a_y=torch.tensor(a_y)
a_y,i=torch.sort(a_y)
one=a_y[a_y==1]
print(one.shape)
print(a_y)
aa_f=[]
for ii in i:

    aa_f.append(a_f[ii])
intermediate_output = np.array(aa_f)  # 100个样本，50个特征
print(intermediate_output.shape)

# 绘制特征热图
plt.figure(figsize=(10, 10))
sns.heatmap(intermediate_output,   cmap='coolwarm', fmt='.2f', linewidths=.5 )
#sns.heatmap(intermediate_output, cmap='viridis', cbar=True)
#sns.heatmap(data, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
#plt.title('Feature heatmap of unsort')
plt.xlabel('Instance Index',fontsize=20)
plt.ylabel('Sample Index',fontsize=20)

plt.savefig("0-100-a_i.jpg")
plt.show()
