import pandas as pd
import numpy as np
import os
from models import LR,FM,MLP,WideAndDeep,DeepFM,FMAndDeep,AFM,NFM,DeepAFM,AutoInt

#data:作者论文 https://github.com/hexiangnan/attentional_factorization_machine

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(2019)
data_dir="../data/movie_lens_afm/"
features=['userId','movieId','tag']

#LIBFM to csv. adapt to alita format.
'''
train=pd.read_csv(data_dir+'ml-tag.train.libfm',sep=' ',names=['y','userId','movieId','tag'])
test=pd.read_csv(data_dir+'ml-tag.test.libfm',sep=' ',names=['y','userId','movieId','tag'])
valid=pd.read_csv(data_dir+'ml-tag.validation.libfm',sep=' ',names=['y','userId','movieId','tag'])

for i,c in enumerate(['userId','movieId','tag']):
    print(i)
    train[c]=train.apply(lambda row:int(row[i+1].split(':')[0]),axis=1)
    test[c] = test.apply(lambda row: int(row[i + 1].split(':')[0]), axis=1)
    valid[c] = valid.apply(lambda row: int(row[i + 1].split(':')[0]), axis=1)

data=pd.concat([train,test,valid])


#data.nunique()
#y              2
#userId     17045
#movieId    23743
#tag        49657
#dtype: int64 符合论文AFM设置.
#The tagging part of the data includes 668, 953 tag applications of 17, 045 users on 23, 743 items with 49, 657 distinct tags. We converted eachtag application (i.e., user ID, movie ID and tag) to a feature vector,resulting in 90, 445 features in total


from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
for c in features:
    data[c]=lbl.fit_transform(list(data[c]))
data.iloc[:train.shape[0],:].to_csv(data_dir+'ml-tag-train.csv',index=False)
data.iloc[train.shape[0]:train.shape[0]+test.shape[0],:].to_csv(data_dir+'ml-tag-test.csv',index=False)
data.iloc[train.shape[0]+test.shape[0]:,:].to_csv(data_dir+'ml-tag-valid.csv',index=False)
'''

train=pd.read_csv(data_dir+'ml-tag-train.csv')
test=pd.read_csv(data_dir+'ml-tag-test.csv')
valid=pd.read_csv(data_dir+'ml-tag-valid.csv')
data=pd.concat([train,test,valid])
features_sizes=[data[c].nunique() for c in features]

y_train=train['y'].values.reshape((-1,1))
y_test=test['y'].values.reshape((-1,1))
y_valid=valid['y'].values.reshape((-1,1))

#lambdas=[0.01,0.1,0.5,1.0,2.0]
#for l in lambdas:
ls=[]
Rounds=1
for _ in range(Rounds):
    model = LR(features_sizes,hash_size=int(1e6)) #valid score 1e5:0.82 3e5:0.79  6e5:0.773 1e6:0.766  | proto test score:0.852
    #model=FM(features_sizes,k=256)#0.474 hash->
    #model = FM(features_sizes, k=24,hash_size=int(1e6)) #protoscore:k=24+h1e6=0.692 0.693 (valid比test好 0.631)
    #model=MLP(features_sizes,deep_layers=(256,256),k=256) #小batch=1024 LR不用小.同1e-3 valid_score=model.fit(train[features],valid[features],y_train,y_valid,lr=0.001,N_EPOCH=100,batch_size=1024,early_stopping_rounds=15)
    #model = DeepFM(features_sizes, deep_layers=(256, 256), k=256)
    #model = NFM(features_sizes, k=256)
    #model = AFM(features_sizes,k=256,attention_FM=256)
    #model = AFM(features_sizes, k=256, attention_FM=8,dropout_keeprate=0.9,lambda_l2=0.001)
    #model = MLP(features_sizes, deep_layers=(1,), k=256)
    #model=AutoInt(features_sizes,k=8)
    valid_score=model.fit(train[features],valid[features],y_train,y_valid,lr=0.001,N_EPOCH=100,batch_size=4096,early_stopping_rounds=15)
    y_pred=model.predict(test[features]).reshape((-1))
    predictions_bounded = np.maximum(y_pred, np.ones(len(y_pred)) * -1)  # bound the lower values
    predictions_bounded = np.minimum(predictions_bounded, np.ones(len(y_pred)) * 1)  # bound the higher values
    test_loss = np.sqrt(np.mean(np.square(y_test.reshape(predictions_bounded.shape) - predictions_bounded)))
    print("Protocol Test Score:",test_loss)
    ls.append(test_loss)

    if _ != Rounds-1:
        model.model.sess.close()
        del model
print(model)
#    print('lambdaL2=',l)
print(" Protocol Test Result : \n%.4f %.4f %s" % (pd.Series(ls).mean(),pd.Series(ls).min(),str(ls)))
