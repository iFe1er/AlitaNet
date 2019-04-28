import pandas as pd
import numpy as np
import os
from models import LR,FM,MLP,WideAndDeep,DeepFM,FMAndDeep,AFM,NFM,DeepAFM



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(2019)
data_dir="../data/movie_lens_20m/"
features=['userId','movieId','tag']

'''
data = pd.read_csv(data_dir+'tags_full.csv')features


print(data.nunique())

#data.nunique()
#userId     19325
#movieId    45981



from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
for c in features:
    data[c]=lbl.fit_transform(list(data[c]))


all_tag=data['tag'].tolist()
np.random.seed(2019)
fake_data=pd.concat([data.copy(),data.copy()]).reset_index(drop=True)
fake_data['tag']=np.random.choice(all_tag,data.shape[0]*2)
#print (data.head())
#print (fake_data.head())

data['y']=1.0
fake_data['y']=-1.0
full_data=pd.concat([data,fake_data]).sample(frac=1.0).reset_index(drop=True)
full_data=full_data.sort_values(['userId','movieId','y'])#(3326991, 4)
#full_data[full_data.duplicated(subset=['userId','movieId','tag'],keep='last')]#可以这样选
full_data=full_data.drop_duplicates(subset=['userId','movieId','tag'],keep='last')#y=1的被keep,剩下(3297759,4)

full_data=full_data.sample(frac=1.0)

print(full_data.head(2))
full_data.to_csv(data_dir+'tag_prediction_latest.csv',index=False)
'''
'''
full_data.head(2)
         userId  movieId    tag    y
1021285    8226     5143  64864 -1.0
2314604   18959      301  74082  1.0
'''


full_data=pd.read_csv(data_dir+'tag_prediction_latest.csv')
features_sizes=[full_data[f].nunique() for f in features]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(full_data[features], full_data['y'], test_size = 0.3, random_state = 42)
y_train=y_train.values.reshape((-1,1))
y_test=y_test.values.reshape((-1,1))
ls=[]
#todo BUG:model.fit会修改train，只能传copy进去
for _ in range(5):
    #model=FM(features_sizes,k=16)
    #model=MLP(features_sizes,deep_layers=(16,16),k=16)
    #model = DeepFM(features_sizes, deep_layers=(16, 16), k=16)
    #model = NFM(features_sizes, k=16)
    model = AFM(features_sizes,k=16,attention_FM=8)

    print(model)
    best_score=model.fit(X_train[features],X_test[features],y_train,y_test,lr=0.0002,N_EPOCH=100,batch_size=1024,early_stopping_rounds=15)
    ls.append(best_score)
print(model)
print("%.4f %.4f %s" % (pd.Series(ls).mean(),pd.Series(ls).min(),str(ls)))

# model=FM(features_sizes,k=16)
# model=MLP(features_sizes,deep_layers=(16,16),k=16)#WideAndDeep(features_sizes, deep_layers=(16, 16), k=16)
# model = DeepFM(features_sizes, deep_layers=(16, 16), k=16)
# model=FMAndDeep(features_sizes, deep_layers=(16, 16), k=16)
# model = FM(features_sizes, k=16, FM_ignore_interaction=[(0, 2), (0, 3), (0, 4)])
# model = DeepFM(features_sizes, deep_layers=(16, 16), k=16, FM_ignore_interaction=[(0, 2), (0, 3), (0, 4)])
# model = FMAndDeep(features_sizes, deep_layers=(16, 16), k=16 , FM_ignore_interaction=[(0,2),(0,3),(0,4)])
# model = AFM(features_sizes,k=16,attention_FM=8)
# model = AFM(features_sizes, k=16, attention_FM=10,FM_ignore_interaction=[(0,2),(0,3),(0,4)])#not that good
# model = NFM(features_sizes, k=16)
# model = DeepAFM(features_sizes,deep_layers=(16, 16), k=16,attention_FM=10)