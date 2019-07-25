import pandas as pd
import numpy as np
import tensorflow as tf
import os
from utils import ColdStartEncoder
from models import LR,FM,MLP,WideAndDeep,DeepFM,FMAndDeep,AFM,CFM,NFM,BiFM,FiBiFM,FiBiNet,DeepAFM,AutoInt,DeepAutoInt,MLR,DCN
from sklearn.metrics import roc_auc_score, log_loss

#DATA:       https://www.kaggle.com/c/amazon-employee-access-challenge/data
#CATBOOST:   https://arxiv.org/pdf/1706.09516.pdf
#BENCHMARK:  https://github.com/catboost/benchmarks/tree/master/quality_benchmarks

data_path='../data/amazon/'
train=pd.read_csv(data_path+'train.csv')
test=pd.read_csv(data_path+'test.csv')

cate_features=train.columns[1:]

features_sizes=train[cate_features].nunique().tolist()

encs=[]
for c in cate_features:
    enc = ColdStartEncoder()
    train.loc[:,c] = enc.fit_transform(train[c])
    test.loc[:,c] = enc.transform(test[c])
    encs.append(enc)

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train[cate_features], train['ACTION'], test_size = 0.2, random_state = 42)
y_train=y_train.values.reshape((-1,1))
y_valid=y_valid.values.reshape((-1,1))

#model=LR(features_sizes,loss_type='binary')
#model=FM(features_sizes,k=8,loss_type='binary')
#model=MLP(features_sizes,k=8,loss_type='binary',deep_layers=(64,32))
#model=NFM(features_sizes,k=8,loss_type='binary')
#model=WideAndDeep(features_sizes,k=8,loss_type='binary',deep_layers=(64,32))
#model=DeepFM(features_sizes,k=8,loss_type='binary',deep_layers=(64,32))
#model=AFM(features_sizes,k=8,loss_type='binary',attention_FM=8)#,lambda_l2=0.005)#oup=1时l2=0.005;oup=4时l2=0.0025
#model=CFM(features_sizes,k=8,loss_type='binary')#,lambda_l2=0.005)#oup=1时l2=0.005;oup=4时l2=0.0025
#model=DeepAFM(features_sizes,k=8,loss_type='binary',attention_FM=8,deep_layers=(64,32))
#model=AutoInt(features_sizes,k=8,loss_type='binary',autoint_params={"autoint_d":16,'autoint_heads':2,"autoint_layers":3,'relu':True,'use_res':True})
#model=DeepAutoInt(features_sizes,k=8,loss_type='binary',deep_layers=(64,32,16),autoint_params={"autoint_d":16,'autoint_heads':2,"autoint_layers":3,'relu':True,'use_res':True})
#model=BiFM(features_sizes,k=8,loss_type='binary')
#model=FiBiFM(features_sizes,k=8,loss_type='binary')
#model=FiBiNet(features_sizes,k=8,loss_type='binary')
#model=DeepBiFM(features_sizes,k=8,loss_type='binary', deep_layers=(64, 32, 16))
#model=DCN(features_sizes,k=8,loss_type='binary',deep_layers=(64,32,16),use_CrossNet_layers=3)

model=MLR(features_sizes,loss_type='binary',MLR_m=4)

best_score = model.fit(X_train[cate_features], X_valid[cate_features], y_train, y_valid, lr=0.001, N_EPOCH=200, batch_size=3277,early_stopping_rounds=5)#0.0005->0.001(1e-3 bs=1000)
y_pred_valid = model.predict(X_valid[cate_features])
y_pred_valid=1./(1.+np.exp(-1.*y_pred_valid))#sigmoid transform
print("Logloss on valid set: %.4f" %log_loss(y_valid,y_pred_valid))


#benchmark
#LGB 0.1536
#Cat:0.1354

#LR bs=3277 0.1555@200   bs=500 0.17591@90  bs=6554 0.1642@200
#FM 0.1490@60 rerun0.1496
#MLP (32,32):0.1545@23  (64,32):0.1539@17 (128,64):0.1552@15 (64,32,16):0.1566@13 (128,64,32):0.1641@11  (64,16):0.1546@16 (48,32):0.1584@24
#NFM 0.1568@19
#WND (64,32) 0.1546@16
#DFM 0.1581  | bs=500 0.1571
#CFM 0.1515@50 rerun0.1540
#AFM:0.1453@34
#DeepAFM (64,32):0.1560 0.1580 | (8,8) 0.1518@34 | (8,4)0.1529@54
#AutoInt d16h2l3:0.1608 | d16h2l2 0.1622 |
#BIFM:0.1502 rerun:0.1557
#FiBiFM:0.1579 0.1571
#DCN(64,32) 0.1593 (16,16)0.1582 (8,8):0.1807
#MLR m=4 0.1493  m=8 0.1465  m=12 0.1469
#rank: Catbst>>AFM>MLR>FM>CFM>LGB>Bifm,WND>LR>DFM>DCN

'''
import lightgbm as lgb
train_Dataset = lgb.Dataset(X_train[cate_features],label=y_train.reshape(-1))
valid_Dataset = lgb.Dataset(X_valid[cate_features],label=y_valid.reshape(-1))
params = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'learning_rate': 0.1 ,
    'verbose': 0,
    'num_leaves': 63,
    'bagging_fraction': 1.0,#0.8 bad
    'bagging_freq': 1,
    'bagging_seed': 1,
    'feature_fraction': 1.0,#0.8 bad
    'feature_fraction_seed': 1,
    'max_bin': 255,
    'max_depth': 15,
    }
bst = lgb.train(params, train_Dataset, num_boost_round=500, valid_sets=[train_Dataset,valid_Dataset],early_stopping_rounds=10)
y_pred_valid = bst.predict(X_valid[cate_features])
print("Logloss on valid set: %.4f" %log_loss(y_valid,y_pred_valid))
'''
TRAIN_CAT=False
if TRAIN_CAT:
    from catboost import CatBoost,Pool
    use_cat_features=True

    cat_features=[i for i in range(len(cate_features))] if use_cat_features else None
    train_pool = Pool(X_train[cate_features],label=y_train.reshape(-1),cat_features=cat_features)
    valid_pool = Pool(X_valid[cate_features],label=y_valid.reshape(-1),cat_features=cat_features)

    param = {'iterations':200,'loss_function':'Logloss','eval_metric':'Logloss','learning_rate':0.15,'depth':6,'logging_level':'Silent','random_seed':0}
    cat_model = CatBoost(param)
    cat_model.fit(train_pool,eval_set=valid_pool,early_stopping_rounds=5)
    y_pred_valid = cat_model.predict(valid_pool, prediction_type='Probability')[:,-1]#y_pred_valid[:,1] #1的概率
    print("Use cat_features:%s. Logloss on valid set:%.4f" %(str(use_cat_features),log_loss(y_valid,y_pred_valid)))
    # catfeatures:0.1354  | no_cat:0.1715
