# 评估不同模型的性能差距

import pandas as pd
from DataFeature import preprocessor
from Kfold import MyKfold

# models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

from EnsembleCLF import EnsembleCLF

# 存储{模型名:[模型,评分,测试集预测结果]}
models = {}


models['LogRegression']=LogisticRegression(max_iter=5000)
models['DecisionTree']=DecisionTreeClassifier(max_depth=7)
models['SVM']=SVC(C=0.5)
models['KNN']=KNeighborsClassifier(n_neighbors=7)
models['GaussBYS']=GaussianNB()
models['AdaBoost']=AdaBoostClassifier(n_estimators=100,learning_rate=0.5)
models['EnsembleCLF(SVM)']=EnsembleCLF(SVC(C=0.5))
models['EnsembleCLF(DT)']=EnsembleCLF(DecisionTreeClassifier(max_depth=5))
models['EnsembleCLF(LR)']=EnsembleCLF(LogisticRegression(max_iter=5000))
models['EnsembleCLF(AdaBoost)']=EnsembleCLF(AdaBoostClassifier())

# 加载数据集
data = pd.read_csv("train.csv")
test_pd = pd.read_csv("test.csv")

# 拆分特征和标签
X_pd = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 归一化处理
X=preprocessor.fit_transform(X_pd)
test=preprocessor.transform(test_pd)
X=pd.DataFrame(X,columns=X_pd.columns)
test=pd.DataFrame(test,columns=test_pd.columns)


# print("ModelName    \t")
res_dict={'Model':[],'Score(10fold)':[]}
predicts=[]
for model_name,model in models.items():
    res_dict['Model'].append(model_name)
    score=MyKfold(model,X,y)
    res_dict['Score(10fold)'].append("{:.3%}".format(score))
    if score > 0.86 :
        predicts.append(model.predict(test))

df=pd.DataFrame(res_dict)
print(df)
print(predicts)
