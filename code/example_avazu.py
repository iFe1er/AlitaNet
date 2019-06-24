import pandas as pd
import numpy as np
import os
from models import LR,FM,MLP,WideAndDeep,DeepFM,FMAndDeep,AFM,NFM,DeepAFM

#data:作者论文 https://github.com/hexiangnan/attentional_factorization_machine

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(2019)
data_dir="../data/avazu/"
features=['id', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
       'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
# #data:    40428967
# #device_id:2686408
# #site_id:  4737
features=['device_id','site_id']

'''
data=pd.read_csv(data_dir+'train.csv')
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
for c in features:
    data[c]=lbl.fit_transform(list(data[c]))
data.to_hdf(data_dir+'train.hdf', 'w',complib='blosc', complevel=5)
'''
data=pd.read_hdf(data_dir+'train.hdf')

features_sizes=[data[c].nunique() for c in features]
#data=data.sample(frac=0.1,random_state=42)
print("Data Prepared.")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[features], data['click'], test_size = 0.2, random_state = 42)
y_train=y_train.values.reshape((-1,1))
y_test=y_test.values.reshape((-1,1))


model=LR(features_sizes,loss_type='binary')
#model=FM(features_sizes,k=10)
# model=MLP(features_sizes,deep_layers=(16,16),k=16)
print(model)
best_score = model.fit(X_train, X_test, y_train, y_test, lr=0.001, N_EPOCH=50, batch_size=6000,early_stopping_rounds=3)#0.0005->0.001(1e-3 bs=1000)

#best_score = model.fit(X_train, X_test, y_train, y_test, lr=0.0002, N_EPOCH=50, batch_size=500,early_stopping_rounds=3)#0.0005->0.001(1e-3 bs=1000)

