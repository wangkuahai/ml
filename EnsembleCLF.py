import numpy as np
import pandas as pd
from sklearn.base import clone

class EnsembleCLF:
    def __init__(self,BaseModel,BaseNum):
        self.BaseModel=BaseModel 
        self.BaseNum=BaseNum
    def fit(self, X_train, y_train):
        self.models=[]
        df = pd.concat([X_train, y_train], axis=1)
        df = df.rename(columns={0: 'label'})
        positive_df = df[df['label'] == 1].copy()  # 正例数据
        negative_df = df[df['label'] == 0].copy()  # 反例数据

        # 统计正例和反例数量
        n_positive = len(positive_df)
        n_negative = len(negative_df)

        # 针对loan问题特化划分 正负例比3:1
        # 计算每份正例的数量
        p = n_positive//n_negative
        self.p=p
        n_per_partition = int(n_positive / p)

        # 随机划分正例
        positive_partitions = []
        for basenum in range(self.BaseNum):
            positive_df=df.sample(frac=1, random_state=42)
            for i in range(int(p)-1):
                partition = positive_df.sample(n_per_partition, replace=False,random_state=11)
                positive_partitions.append(partition)
                positive_df.drop(partition.index, inplace=True)
            positive_partitions.append(positive_df)

        # 组合训练集
        for positive_data in positive_partitions:
            train_data = pd.concat([positive_data, negative_df], axis=0)
            X_train, y_train = train_data.drop('label', axis=1), train_data['label']
            self.models.append(clone(self.BaseModel).fit(X_train,y_train))
    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        votes = np.array(predictions)
        majority_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=votes)
        return majority_vote


