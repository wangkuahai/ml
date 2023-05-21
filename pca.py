import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# 读取数据到Pandas DataFrame
data = pd.read_csv('train.csv')

# 从DataFrame中获取输入特征
X = data.iloc[:, :-1].values

# 执行PCA，提取三个主要特征
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 从DataFrame中获取标签（标签在最后一列）
y = data.iloc[:, -1].values

# 创建颜色映射
cmap = ListedColormap(['red', 'green'])

# 将标签为0的样本设置为红色，标签为1的样本设置为绿色
colors = ['red' if label == 0 else 'green' for label in y]

# 绘制三维散点图
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap=cmap, norm=plt.Normalize(0, 1))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
