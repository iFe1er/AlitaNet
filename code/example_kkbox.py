import pandas as pd
import numpy as np
import tensorflow as tf
import os
from models import LR,FM,MLP,WideAndDeep,DeepFM,FMAndDeep,AFM,NFM,DeepAFM,AutoInt,DeepAutoInt
from sklearn.metrics import roc_auc_score, log_loss

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_path='../data/kkbox/'
train=pd.read_csv(data_path+'train.csv')
members=pd.read_csv(data_path+'members.csv')
songs=pd.read_csv(data_path+'songs.csv')

'''
train.csv: (7377418, 6)
                        UNIQ.          NULL     
msno                   30755  ✓
song_id               359966  ✓
source_system_tab          8  ✓       24849
source_screen_name        20  ✓       414804
source_type               12  ✓       21539
target                     2  Pred_avg:0.504

members.csv:(34403, 7)
msno                      34403 
city                         21 ✓
bd                           95 ✓(cont.)
gender                        2 ✓      19902
registered_via                6 
registration_init_time     3862
expiration_date            1484

songs.csv:(2296320, 7)
song_id        2296320 
song_length     146534(cont.)
genre_ids         1045 ✓               94116
artist_name     222363 ✓
composer        329823                 1071354
lyricist        110925                 1945268
language            10 ✓               1
'''

#[Preprocess]
# Dont Fillnan
#members['gender'] =members['gender'].fillna('unknown')
#songs['genre_ids']=songs['genre_ids'].fillna('-1')
#songs['language']=songs['language'].fillna(-1)

#[Merge]
train=train.merge(members,how='left',on='msno')
train=train.merge(songs,how='left',on='song_id')

#[Features]
id_features=    ['msno','song_id']
member_features=['city','bd','gender']
songs_features= ['genre_ids','artist_name','language','song_length']
context_features=['source_system_tab','source_screen_name','source_type']
features=id_features+member_features+songs_features+context_features

data=train[features+['target']]

#   Fillnan
data.loc[:,'gender'] =    data['gender'].fillna('unknown')            #
data.loc[:,'genre_ids']=  data['genre_ids'].fillna('-1')              #465|458
data.loc[:,'language']=   data['language'].fillna(-1).astype(int)     #52.0->52
data.loc[:,'artist_name']=data['artist_name'].fillna('unknown')       #S.H.E

data.loc[:,'source_system_tab']=data['source_system_tab'].fillna('unknown')
data.loc[:,'source_screen_name']=data['source_screen_name'].fillna('unknown')
data.loc[:,'source_type']=data['source_type'].fillna('unknown')

#print(data.isnull().sum())

'''
#[Test data(to predict)]
test=pd.read_csv(data_path+'test.csv')
test=test.merge(members,how='left',on='msno')
test=test.merge(songs,how='left',on='song_id')
test.loc[:,'gender'] =    test['gender'].fillna('unknown')            #
test.loc[:,'genre_ids']=  test['genre_ids'].fillna('-1')              #465|458
test.loc[:,'language']=   test['language'].fillna(-1).astype(int)     #52.0->52
test.loc[:,'artist_name']=test['artist_name'].fillna('unknown')       #S.H.E
test_data=test[features] #(2556790, 8) test_data=test[train_features] 

test_data[~test_data['msno'].isin(data['msno'].unique())].shape #(184018, 8)个新用户交互
test_data[~test_data['msno'].isin(data['msno'].unique())]['msno'].nunique() #3648个新用户
test_data[~test_data['song_id'].isin(data['song_id'].unique())].shape #(320125, 8)个新歌交互
test_data[~test_data['song_id'].isin(data['song_id'].unique())]['song_id'].nunique() #59873首新歌

train_msno=set(train['msno'].tolist())#30755
test_msno=set(test['msno'].tolist())#25131
intersec_people=len(train_msno&test_msno)#测试的老用户21483 ；新用户=25131-21483=3648 占比17%

train_songs=set(train['song_id'])#359966
test_songs=set(test['song_id'])#224753
intersec_songs=set(train_songs&test_songs)#测试的老歌164880 ；新歌=224753-164880=59873 占比36%

#随机采样技术不能采样出新用户
sample_data=data.sample(frac=0.2,random_state=42)
sample_data_msno=set(sample_data['msno'])#28576
sample_intersec_people=len(train_msno&sample_data_msno)#28576
'''

'''
#padding 特征.副作用.
from utils import multihot_padder
padding_genre,padding_genre_len=multihot_padder(data['genre_ids'])
for i in range(padding_genre_len):
    feature_name='genre_pad_'+str(i+1)
    data[feature_name]=padding_genre[:,i].astype(int)
'''

train_data=data.iloc[:5533063,:].sample(frac=1.0,random_state=42)        #5533063/7377418.0=75%
valid_data=data.iloc[5533063:6270804,:].sample(frac=1.0,random_state=42) #737741/7377418.0 =10%
test_data= data.iloc[6270804:,:].sample(frac=1.0,random_state=42)        #1106614/7377418.0=15%
print("Data Prepared.")


sparse_features=['msno','song_id','city','bd','gender','genre_ids','artist_name','language',
                    'source_system_tab','source_screen_name','source_type']
dense_features=['song_length']
train_features=sparse_features+dense_features
                #['genre_pad_'+str(i+1) for i in range(padding_genre_len)]
print(train_features)

features_sizes=[1+train_data[c].nunique() for c in sparse_features]#todo: 需要+1留出冷启动id
from utils import ColdStartEncoder
encs=[]
for c in sparse_features:
    enc = ColdStartEncoder()#放里面 [BUG fix]
    train_data.loc[:,c] = enc.fit_transform(train_data[c])
    valid_data.loc[:,c] = enc.transform(valid_data[c])
    test_data. loc[:,c] = enc.transform(test_data[c])
    encs.append(enc)

from sklearn.preprocessing import MinMaxScaler,StandardScaler
mns=StandardScaler()#MinMaxScaler(feature_range=(0,1))
train_data.loc[:, dense_features] = mns.fit_transform(train_data[dense_features])
valid_data.loc[:, dense_features] = mns.transform(valid_data[dense_features])
test_data.loc[:, dense_features] = mns.transform(test_data[dense_features])

#end transform

X_train_id,X_valid_id, X_test_id,X_train_dense,X_valid_dense,X_test_dense,y_train, y_valid, y_test = \
                                train_data[sparse_features],valid_data[sparse_features],test_data[sparse_features],\
                                train_data[dense_features],valid_data[dense_features],test_data[dense_features],\
                                                   train_data['target'],valid_data['target'],test_data['target']
y_train=y_train.values.reshape((-1,1))
y_valid=y_valid.values.reshape((-1,1))
y_test =y_test.values.reshape((-1,1))

'''
train_features=['msno','song_id']
features_sizes=[train_data[c].nunique() for c in train_features]
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
enc = ColdStartEncoder()
for c in train_features:
    train_data[c]=lbl.fit_transform(list(train_data[c]))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data[train_features], train_data['target'], test_size = 0.125, random_state = 42)
y_train=y_train.values.reshape((-1,1))
y_test=y_test.values.reshape((-1,1))
'''

#<Model>
#model=LR(features_sizes,loss_type='binary',metric_type='auc')
#model=FM(features_sizes,k=8,loss_type='binary',metric_type='auc')
#model=FM(features_sizes,k=8,loss_type='binary',metric_type='auc',FM_ignore_interaction=[(0,2),(0,3),(0,4)]) #FMDE
#model=MLP(features_sizes,k=8,loss_type='binary',metric_type='auc',deep_layers=(8,8))
#model=NFM(features_sizes,k=8,loss_type='binary',metric_type='auc')
#model=WideAndDeep(features_sizes,k=8,loss_type='binary',metric_type='auc',deep_layers=(8,8))
#model=DeepFM(features_sizes,k=8,loss_type='binary',metric_type='auc',deep_layers=(8,8))
#model=AFM(features_sizes,k=8,loss_type='binary',metric_type='auc',attention_FM=8,lambda_l2=0.005)#oup=1时l2=0.005;oup=4时l2=0.0025
#model=DeepAFM(features_sizes,k=8,loss_type='binary',metric_type='auc',attention_FM=8,deep_layers=(8,8))
#model=AutoInt(features_sizes,k=8,loss_type='binary',metric_type='auc',autoint_params={"autoint_d":16,'autoint_heads':2,"autoint_layers":3,'relu':True,'use_res':True})
#model=DeepAutoInt(features_sizes,k=8,loss_type='binary',metric_type='auc',deep_layers=(24,8),autoint_params={"autoint_d":16,'autoint_heads':2,"autoint_layers":3,'relu':True,'use_res':True})

# +dense model
model=DeepAutoInt(features_sizes,dense_features_size=1,k=8,loss_type='binary',metric_type='auc',deep_layers=(24,8),autoint_params={"autoint_d":16,'autoint_heads':2,"autoint_layers":3,'relu':True,'use_res':True})

print(model)
#[BUG fix] 老版本一定要传入拷贝..wtf~! 06/27修补BUG 内部copy防止影响数据
#best_score = model.fit(X_train_id, X_valid_id, y_train, y_valid, lr=0.0005, N_EPOCH=50, batch_size=4096,early_stopping_rounds=5)#0.0005->0.001(1e-3 bs=1000)
best_score = model.fit(X_train_id, X_valid_id, y_train, y_valid,X_train_dense,X_test_dense, lr=0.0005, N_EPOCH=50, batch_size=4096,early_stopping_rounds=5)#0.0005->0.001(1e-3 bs=1000)

#y_pred_valid = model.predict(X_valid_id)
y_pred_valid = model.predict(X_valid_id,X_valid_dense)
y_pred_valid=1./(1.+np.exp(-1.*y_pred_valid))#sigmoid transform
print("ROC-AUC score on valid set: %.4f" %roc_auc_score(y_valid,y_pred_valid))


y_pred_test=model.predict(X_test_id,X_test_dense)
y_pred_test=1./(1.+np.exp(-1.*y_pred_test))#sigmoid transform
print("ROC-AUC score on test set: %.4f" %roc_auc_score(y_test,y_pred_test))


SUBMIT=False
if SUBMIT:
    test=pd.read_csv(data_path+'test.csv')
    test=test.merge(members,how='left',on='msno')
    test=test.merge(songs,how='left',on='song_id')
    test.loc[:,'gender'] =    test['gender'].fillna('unknown')            #
    test.loc[:,'genre_ids']=  test['genre_ids'].fillna('-1')              #465|458
    test.loc[:,'language']=   test['language'].fillna(-1).astype(int)     #52.0->52
    test.loc[:,'artist_name']=test['artist_name'].fillna('unknown')       #S.H.E
    test.loc[:, 'source_system_tab'] = test['source_system_tab'].fillna('unknown')
    test.loc[:, 'source_screen_name'] = test['source_screen_name'].fillna('unknown')
    test.loc[:, 'source_type'] = test['source_type'].fillna('unknown')
    '''
    padding_genre_test, _ = multihot_padder(test['genre_ids'],padding_len=padding_genre_len)
    for i in range(padding_genre_len):
        feature_name = 'genre_pad_' + str(i + 1)
        test[feature_name] = padding_genre_test[:, i].astype(int)
    '''
    predict_data=test[train_features] #(2556790, 8)
    for i,c in enumerate(train_features):
        enc = encs[i]
        predict_data[c] = enc.transform(predict_data[c])
    y_pred_submit=model.predict(predict_data[train_features])
    y_pred_test=1./(1.+np.exp(-1.*y_pred_submit))#sigmoid transform
    #submission online
    sub=pd.read_csv(data_path+'sample_submission.csv')
    sub['target']=y_pred_test
    sub.to_csv(data_path+'sub/LR_F11_timeSF_valid0.6795_test0.6515.csv',index=False)
    #LR_F19(pad8)_timeSF_valid0.6795_test0.6511.csv
    #AutoInt_d16 L3 H2 RELU_F11_timeSF_valid0.6891_test0.6583.csv

'''
#  msno in test_data & not in train_data
#  set(data['msno'])-(set(test_data['msno'].unique())&set(train_data['msno']))
#  lFUV7lsihiFMPKb+C9EV9w2Y1NsKpgPArWl+Bm7BCCU=

enc=ColdStartEncoder()
enc.fit(train_data['msno'])
tt=enc.transform(test_data['msno'])
#print(tt[test_data['msno']=='lFUV7lsihiFMPKb+C9EV9w2Y1NsKpgPArWl+Bm7BCCU='])  #all zero
'''



'''
import lightgbm as lgb
train_Dataset = lgb.Dataset(X_train,label=y_train.reshape(-1))
valid_Dataset = lgb.Dataset(X_valid,label=y_valid.reshape(-1))
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


y_pred_valid = bst.predict(X_valid)
print("ROC-AUC score on valid set: %.4f" %roc_auc_score(y_valid,y_pred_valid))


y_pred_test=bst.predict(X_test)
print("ROC-AUC score on test set: %.4f" %roc_auc_score(y_test,y_pred_test))

if False:
    test=pd.read_csv(data_path+'test.csv')
    test=test.merge(members,how='left',on='msno')
    test=test.merge(songs,how='left',on='song_id')
    test.loc[:,'gender'] =    test['gender'].fillna('unknown')            #
    test.loc[:,'genre_ids']=  test['genre_ids'].fillna('-1')              #465|458
    test.loc[:,'language']=   test['language'].fillna(-1).astype(int)     #52.0->52
    test.loc[:,'artist_name']=test['artist_name'].fillna('unknown')       #S.H.E
    test.loc[:, 'source_system_tab'] = test['source_system_tab'].fillna('unknown')
    test.loc[:, 'source_screen_name'] = test['source_screen_name'].fillna('unknown')
    test.loc[:, 'source_type'] = test['source_type'].fillna('unknown')
    
    predict_data=test[train_features] #(2556790, 8)
    for i,c in enumerate(train_features):
        enc = encs[i]
        predict_data[c] = enc.transform(predict_data[c])
        
    y_pred_submit=bst.predict(predict_data[train_features])#LGB不需要sigmoid
    #submission online
    sub=pd.read_csv(data_path+'sample_submission.csv')
    sub['target']=y_pred_submit
    sub.to_csv(data_path+'sub/LGB_F11_timeSF_valid0.6470_test0.6339.csv',index=False)
'''