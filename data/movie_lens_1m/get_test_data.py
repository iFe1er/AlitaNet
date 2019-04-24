import pandas as pd

a=pd.read_csv('rating.csv')

#cut 20% data for testing. Every User should cut some.
import numpy as np
np.random.seed(2019)
train_ls,test_ls=[],[]
for uid in range(1,6041):#6041
    if uid%100==0:
        print(uid)
    df=a[a['user_id']==uid]
    test_mask=np.random.choice(len(df),int(0.2*len(df)))
    train_mask=list(set([i for i in range(len(df))])-set(test_mask))
    train_ls.append(df.iloc[train_mask,:])
    test_ls.append(df.iloc[test_mask,:])

train=pd.concat(train_ls).reset_index(drop=True)
test=pd.concat(test_ls).reset_index(drop=True)

train.to_csv('train_rating.csv',index=False)
test.to_csv('test_rating.csv',index=False)