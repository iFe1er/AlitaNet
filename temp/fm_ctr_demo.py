import sys
import numpy as np
import pandas as pd
sys.path.append("../code")
from models import FM

TRAIN=False
if TRAIN:
    x=np.array([
        [0,0,0],
        [0,0,1],
        [1,1,1],
        [1,1,0],
        [2,0,2],
        [2,0,0]
    ])

    y=np.array([1,0,1,0,1,1]).reshape((-1,1))

    x=pd.DataFrame(x,columns=['user_ID','gender','movie_types'])
    features_sizes=x.nunique().tolist()

    model=FM(features_sizes,loss_type='logloss',k=2)
    model.fit(x,x,y,y,lr=0.005,N_EPOCH=200,batch_size=6,early_stopping_rounds=1)
    emb=model.model.embedding_weights.eval(session=model.model.sess)

emb=np.array(
    [[ 0.30337927,  0.36678192],
       [-0.46676683, -0.4239442 ],
       [ 0.9598454 ,  0.96291333],
       [ 1.1335964 ,  0.5938092 ],
       [-1.0082964 , -1.3345039 ],
       [ 1.4716471 ,  0.6463749 ],
       [-1.1967587 , -1.1512431 ],
       [ 0.6090103 ,  0.6722951 ]]
)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

plt.scatter(emb[:,0], emb[:,1])
word_labels=[u'用户1',u'用户2',u'用户3',u'男性',u'女性',u'战争片',u'爱情片',u'卡通片']
for label, x_c, y_c in zip(word_labels, emb[:,0], emb[:,1]):
    plt.annotate(label, xy=(x_c, y_c), xytext=(0, 0), textcoords='offset points')
    plt.xlim(emb[:,0].min()-0.05, emb[:,0].max()+0.05)
    plt.ylim(emb[:,1].min()-0.05, emb[:,1].max()+0.05)
    plt.show()
