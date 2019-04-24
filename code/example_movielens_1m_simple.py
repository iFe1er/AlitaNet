import pandas as pd
import numpy as np
from models import LR,FM,MLP,WideAndDeep,DeepFM,FMAndDeep


if __name__=='__main__':

    np.random.seed(2019)
    data_dir="../data/movie_lens_1m/"
    train = pd.read_csv(data_dir+'train_rating.csv')
    test = pd.read_csv(data_dir+'test_rating.csv')
    data=pd.concat([train,test],axis=0)
    y_train = train['ratings'].values.reshape(-1, 1)  # 一列
    y_test = test['ratings'].values.reshape(-1, 1)


    features=['user_id','movie_id']
    features_sizes=[data[f].nunique() for f in features]
    print(features_sizes)
    print("FM&Deep")

    from sklearn.preprocessing import LabelEncoder
    lbl=LabelEncoder()
    for c in features:
        lbl.fit(list(train[c])+list(test[c]))
        train[c]=lbl.transform(list(train[c]))
        test[c]=lbl.transform(list(test[c]))

    ls=[]
    for _ in range(5):
        #model=FM(features_sizes,k=16)#LR(features_sizes)
        #model=MLP(features_sizes,deep_layers=(16,16),k=16)#WideAndDeep(features_sizes, deep_layers=(16, 16), k=16)
        model = DeepFM(features_sizes, deep_layers=(16, 16), k=16)
        #model=FMAndDeep(features_sizes, deep_layers=(16, 16), k=16)
        best_score=model.fit(train[features],test[features],y_train,y_test,lr=0.0005,N_EPOCH=150,batch_size=500,early_stopping_rounds=20)
        ls.append(best_score)
    print(pd.Series(ls).mean(),pd.Series(ls).min())
    print(str(ls))

