#F(Field)NN+WideAndDeep->EmbdWideAndDeep->DeepFM不再接收
#tf.nn.embedding_lookup : The returned tensor has shape shape(ids) + shape(params)[1:].
import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler
from utils import batcher

class Alita_DeepFM(BaseEstimator):
    # features_sizes: array. number of features in every fields.e.g.[943,1682] user_nunique,movie_nunique
    def __init__(self,features_sizes,loss_type='rmse',k=10,deep_layers=(256,256),activation=tf.nn.relu):

        self.features_sizes=features_sizes
        self.fields=len(features_sizes)
        self.num_features=sum(features_sizes)
        self.loss_type=loss_type
        self.deep_layers=deep_layers
        self.activation=activation
        self.k=k #embedding size K
        initializer=tf.contrib.layers.xavier_initializer()

        self.w=tf.Variable(initializer([self.num_features,1]))
        self.b=tf.Variable(initializer([1]))

        #[Embedding]
        self.embedding_weights=tf.Variable(initializer([self.num_features,k])) # sum_features_sizes,k

        #MLP weights
        self.weights, self.bias = {}, {}
        self.MLP_n_input=k*self.fields
        for i,layer in enumerate(self.deep_layers):
            if i==0:
                self.weights['h'+str(i+1)]=tf.Variable(initializer([self.MLP_n_input,layer]))
            else:
                self.weights['h'+str(i+1)] = tf.Variable(initializer([self.deep_layers[i-1], layer]))
            self.bias['b'+str(i+1)]=tf.Variable(initializer([layer]))
        self.weights['out']= tf.Variable(initializer([self.deep_layers[-1],1]))
        self.bias['out'] = tf.Variable(initializer([1]))

    def _init_session(self):
        return tf.Session()

    def LR(self,ids,w,b):
        #ids:(None,field)  w:(num_features,1)  out:(None,field,1)
        return tf.reduce_sum(tf.reshape(tf.nn.embedding_lookup(w,ids),[-1,self.fields]),axis=1)+b

    def Embedding(self,ids,params):
        #params:self.embedding(sum_features_sizes,k)   ids:(None,fields)  out=shape(ids)+shape(params[1:])=(None,fields,k)
        return tf.nn.embedding_lookup(params,ids)

    def MLP(self,x,weights,bias):
        last_layer=None
        for i,layer in enumerate(self.deep_layers):
            if i==0:
                last_layer=self.activation(tf.matmul(x,weights['h1'])+bias['b1'])
            else:
                this_layer=self.activation(tf.matmul(last_layer, weights['h'+str(i+1)]) + bias['b'+str(i+1)])
                last_layer=this_layer
        return tf.matmul(last_layer,weights['out'])+bias['out']

    #每个不同field交叉内积的和
    def FM(self,embedding):#embedding:(None,field,k)
        assert self.fields>=2,"Must have more than 2 fields to do FM"
        cross_term=0
        for i in range(self.fields):
            for j in range(i+1,self.fields):
                #embedding[:,i,:] shape是(None,k)
                cross_term=cross_term+tf.reduce_sum(embedding[:,i,:]*embedding[:,j,:],axis=1,keepdims=True)#(None,k)->(None,1)
        return cross_term

    # 每个不同field交叉内积的和，等价于(每个field的和的平方减去每个field的平方的和)/2  ，因为（a+b+c）²=a²+b²+c²+2ab+2ac+2bc.
    def FM2(self,embedding):#embedding:(None,field,k)
        assert self.fields>=2,"Must have more than 2 fields to do FM"
        square_sum=tf.square(tf.reduce_sum(embedding,axis=1))#(None,k)
        sum_square=tf.reduce_sum(tf.square(embedding),axis=1)#(None,k)
        cross_term=0.5*tf.reduce_sum(square_sum-sum_square,axis=1,keepdims=True)#(None,1)
        return cross_term

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        #data preprocess:对ids的每个features，label encoder都要从上一个的末尾开始。函数输入时则保证每个都从0起.
        for i,column in enumerate(ids_train.columns):
            if i>=1:
                ids_train.loc[:,column]=ids_train[column]+sum(self.features_sizes[:i])
                ids_test.loc[:, column]=ids_test[column]+sum(self.features_sizes[:i])
        self.ids=tf.placeholder(tf.int32,[None,self.fields])
        self.y=tf.placeholder(tf.float32,[None,1])

        self.embedding=self.Embedding(self.ids,self.embedding_weights)#(None,fields,k)
        MLP_in=tf.reshape(self.embedding,[-1,self.fields*self.k])

        self.pred=self.MLP(MLP_in,self.weights,self.bias)+self.FM2(self.embedding)+self.LR(self.ids,self.w,self.b)
        if self.loss_type=='rmse':
            self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.pred)))
        elif self.loss_type=='mse':
            self.loss = tf.reduce_mean(tf.square(self.y-self.pred))
        elif self.loss_type=='binary_crossentropy':
            self.loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,logits=self.pred))
        else:
            raise Exception("Loss type %s not supported"%self.loss_type)

        self.optimizer=tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.sess=self._init_session()
        self.sess.run(tf.global_variables_initializer())

        cur_best_rounds=0
        cur_min_loss=1e8
        best_weights = {v.name: v.eval(self.sess) for v in tf.trainable_variables()}
        for epoch in range(N_EPOCH):
            train_loss=0.
            total_batches=int(ids_train.shape[0]/batch_size)
            for bx,by in batcher(ids_train,y_train,batch_size):
                _,l=self.sess.run([self.optimizer,self.loss],feed_dict={self.ids:bx,self.y:by})
                train_loss+=l
            train_loss/=total_batches
            test_loss=self.sess.run(self.loss,feed_dict={self.ids:ids_test,self.y:y_test})
            print("epoch:%s train_loss:%s test_loss:%s" %(epoch+1,train_loss,test_loss))
            #print("self.pred=",self.sess.run(self.pred,feed_dict={self.ids:ids_test,self.y:y_test}))
            #print("self.y=",y_test)
            if test_loss<cur_min_loss:
                cur_min_loss=test_loss
                cur_best_rounds=epoch+1
                best_weights = {v.name: v.eval(self.sess) for v in tf.trainable_variables()}
            if epoch+1-cur_best_rounds>=early_stopping_rounds:
                print("Early Stopping because not improved for %s rounds" % early_stopping_rounds)
                self.sess.run(tf.tuple([tf.assign(var, best_weights[var.name]) for var in tf.trainable_variables()]))
                best_score = self.sess.run(self.loss, feed_dict={self.ids: ids_test, self.y: y_test, })
                print("Best Score:",best_score,' at round ',cur_best_rounds)
                return best_score
        self.sess.run(tf.tuple([tf.assign(var, best_weights[var.name]) for var in tf.trainable_variables()]))
        best_score=self.sess.run(self.loss, feed_dict={self.ids: ids_test, self.y: y_test,})
        print("Best Score:", best_score,' at round ',cur_best_rounds)
        return best_score


    def predict(self,ids_pred):
        for i,column in enumerate(ids_pred.columns):
            if i>=1:
                ids_pred.loc[:,column]=ids_pred[column]+sum(self.features_sizes[:i])
        self.output=self.sess.run(self.pred,feed_dict={self.ids: ids_pred,})
        return self.output


if __name__ == '__main__':
    import pandas as pd
    np.random.seed(2019)
    data_dir="../data/movie_lens_1k/"
    train = pd.read_csv(data_dir+'ua.base', sep='\t', names=['user_id', 'movie_id', 'ratings', 'time'])
    test = pd.read_csv(data_dir+'ua.test', sep='\t', names=['user_id', 'movie_id', 'ratings', 'time'])
    data=pd.concat([train,test],axis=0)
    y_train = train['ratings'].values.reshape(-1, 1)  # 一列
    y_test = test['ratings'].values.reshape(-1, 1)


    features=['user_id','movie_id']
    features_sizes=[data[f].nunique() for f in features]
    print("Deep+FM+LR_")
    ls=[]
    for _ in range(10):
        ELR=Alita_DeepFM(features_sizes,deep_layers=(10,10))
        best_score=ELR.fit(train[features]-1,test[features]-1,y_train,y_test,lr=0.0005,N_EPOCH=150,batch_size=500,early_stopping_rounds=30)
        #-1是因为ids要从0起.而数据中是从1起的
        ls.append(best_score)
    print(pd.Series(ls).mean(),pd.Series(ls).min())
    print(str(ls))