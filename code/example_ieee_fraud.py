import pandas as pd
import numpy as np
import tensorflow as tf
import os
from models import LR,FM,MLP,WideAndDeep,DeepFM,FMAndDeep,AFM,CFM,NFM,BiFM,FiBiFM,FiBiNet,DeepAFM,AutoInt,DeepAutoInt,MLR
from sklearn.metrics import roc_auc_score, log_loss


#https://www.kaggle.com/c/ieee-fraud-detection/data
data_path='../data/ieee_fraud_detection/'
train_transaction = pd.read_csv(data_path+'train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv(data_path+'test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv(data_path+'train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv(data_path+'test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv(data_path+'sample_submission.csv')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

#y_train = train['isFraud'].copy()
del train_transaction, train_identity, test_transaction, test_identity


print(train.shape)
print(test.shape)

# Label Encoding
from sklearn.preprocessing import LabelEncoder
from utils import ColdStartEncoder

cate_features=['ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','P_emaildomain','R_emaildomain',
            'M1','M2','M3','M4','M5','M6','M7','M8','M9','DeviceType','DeviceInfo']+['id_'+str(i) for i in range(12,39)]#49  减掉sparse_features刚好是18

for f in cate_features:
    if train[f].dtype=='float':
        train.loc[:,f]=train[f].fillna(-999).astype(int)
        test.loc[:,f]=test[f].fillna(-999).astype(int)
    else:
        train.loc[:,f] = train[f].fillna('UKN')
        test.loc[:,f] = test[f].fillna('UKN')

features_sizes=[train[f].nunique() for f in cate_features]

encs=[]
for c in cate_features:
    print(c)
    enc = ColdStartEncoder()
    train.loc[:,c] = enc.fit_transform(train[c])
    test.loc[:,c] = enc.transform(test[c])
    encs.append(enc)

train_transaction_id,test_transaction_id=train.index,test.index
train=train.reset_index(drop=True)
test=test.reset_index(drop=True)
train_y=train['isFraud'].reset_index(drop=True)

X_train,X_valid=train.iloc[:472432,:].sample(frac=1.0,random_state=42),\
                train.iloc[472432:,:].sample(frac=1.0,random_state=42)
y_train=X_train['isFraud'].values.reshape((-1,1))
y_valid=X_valid['isFraud'].values.reshape((-1,1))

#model=LR(features_sizes,loss_type='binary',metric_type='auc')
#model=FM(features_sizes,k=8,loss_type='binary',metric_type='auc')
#model=MLP(features_sizes,k=8,loss_type='binary',metric_type='auc',deep_layers=(32,32))
#model=BiFM(features_sizes,k=8,loss_type='binary',metric_type='auc')
#model=DeepFM(features_sizes,k=8,loss_type='binary',metric_type='auc',deep_layers=(32,32))
#model=AFM(features_sizes,loss_type='binary',metric_type='auc',attention_FM=8)
model=CFM(features_sizes,loss_type='binary',metric_type='auc')
#model=MLR(features_sizes,loss_type='binary',metric_type='auc',MLR_m=16)
best_score = model.fit(X_train[cate_features], X_valid[cate_features], y_train, y_valid, lr=0.0005, N_EPOCH=50, batch_size=500,early_stopping_rounds=3)#0.0005->0.001(1e-3 bs=1000)

SUBMIT=False
if SUBMIT:
    y_pred=model.predict(test[cate_features])
    y_pred = 1. / (1. + np.exp(-1. * y_pred))
    sample_submission['isFraud'] = y_pred
    #sample_submission.to_csv(data_path+'sub/sub01_LR_F49_timeSF_0.8154.csv',index=False)
    #sample_submission.to_csv(data_path+'sub/sub05_MLR_m=15_nosig_F49_timeSF_0.8154.csv',index=False)

#LR:0.8774 KG:0.8261
#TIMESF
# LGB:0.8442@90 KG=0.8549
# CAT:with cat_fea
#    dep=6:0.8471
#    dep=8:0.8674 kg=0.8597
# LR:0.8154@11 KG=0.8468
# FM:0.8220@3  KG=0.8384  | FM k=6 0.8225  | FM k=4 0.8109 | k=3 0.8133
# MLP(256,128,64):0.8066@1 KG=  | (64,64) 0.8022@1
# BiFM 0.8136  (very slow when fields a lot)
# WND  0.8037@1
# DeepFM 0.8083@1
# MLR m=2: with sig:0.67   no sig:0.8093@2  | m=4 no sig 0.8061@10 KG=0.8378 |m=12 nosig 0.8096@1 KG=0.8472 |m=15 nosig 0.8117@2 KG=0.8381
# AFM:0.8157@1 KG=0.8452
# CFM:0.8142@1 KG=0.8425

#KG rank: CATbst>LGB>MLR>LR>AFM>FM

'''
import lightgbm as lgb
train_Dataset = lgb.Dataset(X_train[cate_features],label=y_train.reshape(-1))
valid_Dataset = lgb.Dataset(X_valid[cate_features],label=y_valid.reshape(-1))
params = {
    'objective': 'binary',
    'metric': 'auc',
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
print("ROC-AUC score on valid set: %.4f" %roc_auc_score(y_valid,y_pred_valid))

y_pred=bst.predict(test[cate_features])
sample_submission['isFraud'] = y_pred
sample_submission.to_csv(data_path+'sub/sub04_LGB_F49_timeSF_0.8442.csv',index=False)
'''

TRAIN_CAT=False
if TRAIN_CAT:
    from catboost import CatBoost, Pool
    use_cat_features = True
    cat_features = [i for i in range(len(cate_features))] if use_cat_features else None
    train_pool = Pool(X_train[cate_features], label=y_train.reshape(-1), cat_features=cat_features)
    valid_pool = Pool(X_valid[cate_features], label=y_valid.reshape(-1), cat_features=cat_features)

    param = {'iterations': 200, 'loss_function': 'Logloss', 'eval_metric': 'AUC', 'learning_rate': 0.1, 'depth': 6,
             'logging_level': 'Verbose', 'random_seed': 0}
    cat_model = CatBoost(param)
    cat_model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=5)
    y_pred_valid = cat_model.predict(valid_pool, prediction_type='Probability')[:, -1]  # y_pred_valid[:,1] #1的概率
    print("Use cat_features:%s. AUC on valid set:%.4f" % (str(use_cat_features), roc_auc_score(y_valid, y_pred_valid)))

    y_pred=cat_model.predict(test[cate_features])
    sample_submission['isFraud'] = y_pred
    sample_submission.to_csv(data_path+'sub/sub_CAT_dep8_F49_timeSF_0.8674.csv',index=False)


'''
from sklearn.model_selection import KFold
kf=KFold(n_splits=3,shuffle=True,random_state=42)
for train_index, test_index in kf.split(train):
    X_train, X_valid = train.loc[train_index,cate_features], train.loc[test_index,cate_features]
    y_train, y_valid = train_y.loc[train_index].values.reshape((-1,1)), train_y.loc[test_index].values.reshape((-1,1))
    break
'''