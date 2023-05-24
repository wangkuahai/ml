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
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier

from sklearn.metrics import accuracy_score,f1_score


from EnsembleCLF import EnsembleCLF

# 存储{模型名:[模型,评分,测试集预测结果]}
models = {}

oracle=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

models['LogRegression']=LogisticRegression(max_iter=5000)
models['DecisionTree']=DecisionTreeClassifier(max_depth=7)
models['SVM']=SVC(C=0.5)
models['KNN']=KNeighborsClassifier(n_neighbors=7)
models['GaussBYS']=GaussianNB()
models['AdaBoost']=AdaBoostClassifier(n_estimators=100,learning_rate=0.5)
models['EnsembleCLF(SVM)']=EnsembleCLF(SVC(C=0.5),3)
models['EnsembleCLF(DT)']=EnsembleCLF(DecisionTreeClassifier(max_depth=5),3)
models['EnsembleCLF(LR)']=EnsembleCLF(LogisticRegression(max_iter=5000),3)
models['EnsembleCLF(AdaBoost)']=EnsembleCLF(AdaBoostClassifier(),3)
models['EnsembleCLF(XGBoost)']=EnsembleCLF(XGBClassifier(),3)
models['XGBoost']=XGBClassifier(n_estimators=50,random_state=21,learning_rate=0.3)
models['XGBoostRF']=XGBRFClassifier()
models['RandomForest']=RandomForestClassifier(n_estimators=500)

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
res_dict={'Model':[],'Score(5fold)':[],'F1Score(5fold)':[],'OrcaleAC':[],'OrcaleF1':[],'Result':[]}
for model_name,model in models.items():
    res_dict['Model'].append(model_name)
    score,f1=MyKfold(model,X,y)
    res_dict['Score(5fold)'].append("{:.3%}".format(score))
    res_dict['F1Score(5fold)'].append("{:.3%}".format(f1))
    predict=model.predict(test)
    res_dict['Result'].append(predict)
    
    res_dict['OrcaleAC'].append(accuracy_score(oracle,predict))
    res_dict['OrcaleF1'].append(f1_score(oracle,predict,average='macro'))


df=pd.DataFrame(res_dict)
print(list(df['Result']))

print(df)
