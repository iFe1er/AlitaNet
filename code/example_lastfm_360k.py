import pandas as pd
import numpy as np
import os
from models import LR,FM,MLP,WideAndDeep,DeepFM,FMAndDeep,AFM,NFM,DeepAFM

#data:作者论文 https://github.com/hexiangnan/attentional_factorization_machine

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

interact=pd.read_csv('../data/last_fm_360K/usersha1-artmbid-artname-plays.tsv',sep='\t',names=['userId','artistId','artistName','plays']).dropna()
user_profile=pd.read_csv('../data/last_fm_360K/usersha1-profile.tsv',sep='\t',names=['userId','gender','age','country','signupDate'])
user_profile['age']=user_profile['age'].fillna('-1')
user_profile['gender']=user_profile['gender'].fillna('n')

#merge
data=interact.drop('artistName',axis=1).merge(user_profile.drop('signupDate',axis=1),how='left',on='userId').dropna()# #85个nan
print("Data Prepared.")

#log1p process on plays. loss=rmsle
#sns.distplot(np.log1p(data['plays']),bins=100)

#features=['userId', 'artistId', 'gender', 'age', 'country']
features=['userId', 'artistId']

features_sizes=[data[c].nunique() for c in features]
data['plays']=np.log1p(data['plays'])



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[features], data['plays'], test_size = 0.2, random_state = 42)
y_train=y_train.values.reshape((-1,1))
y_test=y_test.values.reshape((-1,1))


model=LR(features_sizes,loss_type='binary',hash_size=100000)
#model=FM(features_sizes,k=10)
# model=MLP(features_sizes,deep_layers=(16,16),k=16)
print(model)
best_score = model.fit(X_train, X_test, y_train, y_test, lr=0.001, N_EPOCH=50, batch_size=5000,early_stopping_rounds=3)#0.0005->0.001(1e-3 bs=1000)
