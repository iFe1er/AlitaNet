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
data['plays']=np.log1p(data['plays'])

print("Data Prepared.")

#log1p process on plays. loss=rmsle
#sns.distplot(np.log1p(data['plays']),bins=100)

#features=['userId', 'artistId', 'gender', 'age', 'country']
features=['userId', 'artistId','gender', 'age', 'country']

features_sizes=[data[c].nunique() for c in features]
#[358856, 160111]
#[358856, 160111, 3, 114, 239]


from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
for c in features:
    data[c]=lbl.fit_transform(list(data[c]))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[features], data['plays'], test_size = 0.2, random_state = 42)
y_train=y_train.values.reshape((-1,1))
y_test=y_test.values.reshape((-1,1))


#model=LR(features_sizes,loss_type='rmse')#,hash_size=r)
#model=FM(features_sizes,k=24)
#model=MLP(features_sizes,deep_layers=(12,12),k=24) best(12,12) k=24
#model=FM(features_sizes,k=24,FM_ignore_interaction=[(0,2),(0,3),(0,4)])
model=FM(features_sizes,k=24,FM_ignore_interaction=[(0,1),(0,2),(0,3),(0,4)])
#model=DeepFM(features_sizes,deep_layers=(12,12),k=24)
#model = NFM(features_sizes, k=24)
print(model)
best_score = model.fit(X_train, X_test, y_train, y_test, lr=0.0005, N_EPOCH=50, batch_size=5000,early_stopping_rounds=5)#0.0005->0.001(1e-3 bs=1000)

'''
ls=[]
Rounds=1
for _ in range(Rounds):
        ls.append(best_score)
print(model)
print(" Protocol Test Result : \n%.4f %.4f %s" % (pd.Series(ls).mean(),pd.Series(ls).min(),str([round(i,4) for i in ls])))
'''