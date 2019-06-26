import pandas as pd
import numpy as np
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
test_data=test[features]

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
#测试占比20% ，新用户再占20%  1230个新用户
test_msno=np.random.choice(train['msno'].unique(),int(train['msno'].nunique()*0.2*0.2))

test_data_new=data[data['msno'].isin(test_msno)]#279964=28W
test_data_old=data[~data['msno'].isin(test_msno)].sample(int(data.shape[0]*0.2*0.8),random_state=42)#1180386
test_data=pd.concat([test_data_new,test_data_old],axis=0)#1460350