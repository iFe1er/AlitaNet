import pandas as pd
import numpy as np
from models import LR,FM,MLP,WideAndDeep,DeepFM


if __name__=='__main__':

    np.random.seed(2019)
    data_dir="../data/movie_lens_100k/"
    train = pd.read_csv(data_dir+'ua.base', sep='\t', names=['user_id', 'movie_id', 'ratings', 'time'])
    test = pd.read_csv(data_dir+'ua.test', sep='\t', names=['user_id', 'movie_id', 'ratings', 'time'])
    data=pd.concat([train,test],axis=0)
    y_train = train['ratings'].values.reshape(-1, 1)  # 一列
    y_test = test['ratings'].values.reshape(-1, 1)


    features=['user_id','movie_id']
    features_sizes=[data[f].nunique() for f in features]
    print("DFM")
    ls=[]
    model=LR
    for _ in range(10):
        model=FM(features_sizes)
        #model = LR(features_sizes)
        #model=DeepFM(features_sizes,deep_layers=(10,10),k=10)
        best_score=model.fit(train[features]-1,test[features]-1,y_train,y_test,lr=0.0005,N_EPOCH=150,batch_size=500,early_stopping_rounds=30)
        #-1是因为ids要从0起.而数据中是从1起的
        ls.append(best_score)
    print(pd.Series(ls).mean(),pd.Series(ls).min())
    print(str(ls))