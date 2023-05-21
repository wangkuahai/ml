import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

def MyKfold(model, X, y):
    # 进行数据预处理 对于特征相同标签不同的样例，在留出法测试的时候，选择标签比重大的作为最终标签，但训练的时候不进行处理
    df = X
    result_dict={}
    # 遍历每一行，将值存储到字典中
    for index,row in df.iterrows():
        key = tuple(row)
        value = y.iloc[index]
        if key in result_dict:
            if value in result_dict[key]:
                result_dict[key][value] += 1
            else:
                result_dict[key][value] = 1
        else:
            result_dict[key] = {value: 1}

    # 实现10折交叉验证
    scores = []  # 存储每个 fold 的准确率
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, val_idx in kfold.split(X, y):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        
        y_pred = model.predict(X_val_fold)

        # 从result_dict中查询y_val_fold
        for index,x_val in X_val_fold.iterrows():
            key = tuple(x_val) 
            if len(result_dict[key]) == 1:
                y_val_fold.at[index]=next(iter(result_dict[key]))
            elif result_dict[key][0] > result_dict[key][1] :
                y_val_fold.at[index]=0
            else:
                y_val_fold.at[index]=1

        score = accuracy_score(y_val_fold, y_pred)
        scores.append(score)
        # print(model.predict(test))

    # 计算平均准确率并返回
    avg_score = sum(scores) / len(scores)
    return avg_score

