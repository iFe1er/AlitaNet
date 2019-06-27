import pandas as pd
import numpy as np
import tensorflow as tf
import os
from models import LR,FM,MLP,WideAndDeep,DeepFM,FMAndDeep,AFM,NFM,DeepAFM

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
source_system_tab          8           24849
source_screen_name        20           414804
source_type               12           21539
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
songs_features= ['genre_ids','artist_name','language']
features=id_features+member_features+songs_features

data=train[features+['target']]

#   Fillnan
data.loc[:,'gender'] =    data['gender'].fillna('unknown')            #
data.loc[:,'genre_ids']=  data['genre_ids'].fillna('-1')              #465|458
data.loc[:,'language']=   data['language'].fillna(-1).astype(int)     #52.0->52
data.loc[:,'artist_name']=data['artist_name'].fillna('unknown')       #S.H.E

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
test_data=test[features] #(2556790, 8)

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



#同样比例获取测试集msnoID
np.random.seed(42)
#测试占比20% ，给3000个新用户
test_msno=np.random.choice(train['msno'].unique(),2000)
test_song=np.random.choice(train['song_id'].unique(),20000)

test_data_new_msno=data[data['msno'].isin(test_msno)]#448194
test_data_new_song=data[(~data['msno'].isin(test_msno))&(data['song_id'].isin(test_song))]#415334

test_data_old=data[(~data['msno'].isin(test_msno))&(~data['song_id'].isin(test_song))].sample(frac=0.10,random_state=42)#651389
test_data=pd.concat([test_data_new_msno,test_data_new_song,test_data_old],axis=0)#1514917
#train_data=data.loc[data.index.difference(test_data.index),:]#5862501

'''
train_data
msno            28619
song_id        314735
city               21
bd                 92
gender              3
genre_ids         546
artist_name     37336
language           10
target              2
'''

print("Data Prepared.")
train_data=data.loc[data.index.difference(test_data.index),:]#5862501

train_features=['msno','song_id']
features_sizes=[train_data[c].nunique() for c in train_features]

from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
for c in train_features:
    train_data[c]=lbl.fit_transform(list(train_data[c]))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data[train_features], train_data['target'], test_size = 0.125, random_state = 42)
y_train=y_train.values.reshape((-1,1))
y_test=y_test.values.reshape((-1,1))

#<Model>
#model=LR(features_sizes,loss_type='binary',metric_type='auc')
model=FM(features_sizes,k=8,loss_type='binary',metric_type='auc')
print(model)
#[BUG fix] 老版本一定要传入拷贝..wtf~! 06/27修补BUG 内部copy防止影响数据
best_score = model.fit(X_train[train_features], X_test[train_features], y_train, y_test, lr=0.0005, N_EPOCH=50, batch_size=4096,early_stopping_rounds=5)#0.0005->0.001(1e-3 bs=1000)
y_pred=model.predict(X_test)
y_pred=1./(1.+np.exp(-1.*y_pred))#sigmoid transform
from sklearn.metrics import roc_auc_score,log_loss
print("ROC-AUC score on valid set: %.4f" %roc_auc_score(y_test,y_pred))
#print(log_loss(y_test,y_pred))
