import pandas as pd
import numpy as np
from models import LR,FM,MLP,WideAndDeep,DeepFM,FMAndDeep,AFM


if __name__=='__main__':

    np.random.seed(2019)
    data_dir="../data/movie_lens_1m/"
    train = pd.read_csv(data_dir+'train_rating.csv')
    test = pd.read_csv(data_dir+'test_rating.csv')

    y_train = train['ratings'].values.reshape(-1, 1)  # 一列
    y_test = test['ratings'].values.reshape(-1, 1)

    train=train.merge(pd.read_csv(data_dir+'user.csv')[['user_id','gender','age','occupation']],on='user_id')
    test=test.merge(pd.read_csv(data_dir+'user.csv')[['user_id','gender','age','occupation']],on='user_id')

    #train=train.merge(pd.read_csv(data_dir+'movie.csv')[['movie_id','genres']],on='movie_id')
    #test=test.merge(pd.read_csv(data_dir+'movie.csv')[['movie_id','genres']],on='movie_id')

    print(train.columns)
    features=['user_id','movie_id','gender','age','occupation']#,'genres']

    data = pd.concat([train, test], axis=0)
    features_sizes=[data[f].nunique() for f in features]
    print(features_sizes)


    from sklearn.preprocessing import LabelEncoder
    lbl=LabelEncoder()
    for c in features:
        lbl.fit(list(train[c])+list(test[c]))
        train[c]=lbl.transform(list(train[c]))
        test[c]=lbl.transform(list(test[c]))

    ls=[]
    for _ in range(5):
        #model=LR(features_sizes)
        #model=FM(features_sizes,k=16)
        #model=MLP(features_sizes,deep_layers=(16,16),k=16)#WideAndDeep(features_sizes, deep_layers=(16, 16), k=16)
        #model = DeepFM(features_sizes, deep_layers=(16, 16), k=16)
        #model=FMAndDeep(features_sizes, deep_layers=(16, 16), k=16)
        #model = FM(features_sizes, k=16, FM_ignore_interaction=[(0, 2), (0, 3), (0, 4)])
        #model = DeepFM(features_sizes, deep_layers=(16, 16), k=16, FM_ignore_interaction=[(0, 2), (0, 3), (0, 4)])
        #model = FMAndDeep(features_sizes, deep_layers=(16, 16), k=16 , FM_ignore_interaction=[(0,2),(0,3),(0,4)])
        model = AFM(features_sizes,k=16,attention_FM=10)
        #model = AFM(features_sizes, k=16, attention_FM=10,FM_ignore_interaction=[(0,2),(0,3),(0,4)])#not that good
        print(model)
        best_score=model.fit(train[features],test[features],y_train,y_test,lr=0.0005,N_EPOCH=150,batch_size=2000,early_stopping_rounds=20)
        ls.append(best_score)
    print(model)
    print(pd.Series(ls).mean(),pd.Series(ls).min())
    print(str(ls))

    '''
    #observe AFM attention mask.
    t = model.model.get_attention_mask()
    tt=np.vstack(t)#tt.shape=(197656, 10, 1)
    print(tt.mean(axis=0))
    '''