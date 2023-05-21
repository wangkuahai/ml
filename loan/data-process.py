import pandas as pd 
import sys
sys.path.append("../tools")
from deal_column import drop,oneHot

print("Processing [loan] data...")

def deal_csv(path,train=False):
    df=pd.read_csv(path).iloc[:,2:]
    
    # 处理 due_date
    df=drop(df,'due_date')

    # 处理 education 转化为连续整型，保留序关系
    education_map = {'High School or Below': 0,
                 'college': 1,
                 'Bechalor': 2,
                 'Master or Above': 3}
    df['education'] = df['education'].replace(education_map)
    
    # 处理  effective_date , drop
    # df= oneHot(df,'effective_date')
    df=drop(df,'effective_date')

    # 处理 Gender 映射到0 1
    gender_map= {'male': 1, 'female':0}
    df['Gender']=df['Gender'].replace(gender_map)

    # 处理标签 loan_status
    if train :
        label_map= {'PAIDOFF': 1, 'COLLECTION':0}
        df['loan_status'] = df['loan_status'].replace(label_map)
        label = df.pop('loan_status')
        df.insert(len(df.columns),'label',label)
    df.to_csv('../{}'.format(path),index=False)

deal_csv('train.csv',True)
deal_csv('test.csv')
