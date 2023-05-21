import pandas as pd 

def drop(df,column_name):
    df=df.drop(column_name,axis=1)
    return df
def oneHot(df,column_name):
    one_hot = pd.get_dummies(df[column_name],dtype=int)
    df = pd.concat([df, one_hot], axis=1)
    df = df.drop(column_name, axis=1)
    return df
