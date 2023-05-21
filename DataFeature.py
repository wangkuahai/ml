
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# 定义连续特征的索引
continuous_features = [0, 1, 2, 3 , 4]
dis_features = []
# 定义预处理方法
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_features),
        ('cat', 'passthrough', dis_features)  # 保留分类特征
    ])
