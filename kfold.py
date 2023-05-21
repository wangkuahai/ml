import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, KFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

scaler = StandardScaler()
# 加载数据集
data = pd.read_csv("train.csv")
test_pd = pd.read_csv("test.csv")

# 拆分特征和标签
X_pd = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 定义连续特征的索引
continuous_features = [0, 1, 2, 3]
dis_features = [4,5,6,7,8,9,10,11]

# 定义预处理方法
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_features),
        ('cat', 'passthrough', dis_features)  # 保留分类特征
    ])

# 归一化处理
X=preprocessor.fit_transform(X_pd)
test=preprocessor.transform(test_pd)
X=pd.DataFrame(X,columns=X_pd.columns)
test=pd.DataFrame(test,columns=test_pd.columns)


# 构建 RandomUnderSampler 和 RandomOverSampler 对象
rus = RandomOverSampler(random_state=42)

# 实现10折交叉验证
scores = []  # 存储每个 fold 的准确率
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
for train_idx, val_idx in kfold.split(X, y):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

    # print(train_idx,val_idx)

    # 对数据集进行类别平衡处理
    # X_train_fold, y_train_fold = rus.fit_resample(X_train_fold, y_train_fold) # RandomUnderSampler

    # 创建逻辑回归模型，并在每个 fold 上训练模型，并在验证集上进行评估
    # model = SVC(C=0.5) # 0.75
    # model = LogisticRegression(max_iter=5000) # 0.71
    # model = DecisionTreeClassifier(criterion='gini',random_state=1)
    # model= KNeighborsClassifier(n_neighbors=5,p=2,metric="minkowski")
    # model = GaussianNB()

    model = BaggingClassifier(estimator=SVC(C=0.5), n_estimators=10, random_state=0)


    model.fit(X_train_fold, y_train_fold)
    
    y_pred = model.predict(X_val_fold)
    score = accuracy_score(y_val_fold, y_pred)
    scores.append(score)
    # print(model.predict(test))

# 计算平均准确率并输出
avg_score = sum(scores) / len(scores)
print('Average score:', avg_score)

model.fit(X,y)
print(model.predict(test))
