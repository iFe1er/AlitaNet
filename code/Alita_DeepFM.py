import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator
from utils import batcher

class Alita_DeepFM(BaseEstimator):
    # features_sizes: array. number of features in every fields.e.g.[943,1682] user_nunique,movie_nunique
    def __init__(self,features_sizes,loss_type='rmse',k=10,deep_layers=(256,256),activation=tf.nn.relu,use_LR=True,use_FM=True,use_MLP=True,FM_ignore_interaction=None,attention_FM=0,use_NFM=False):
        self.features_sizes=features_sizes
        self.fields=len(features_sizes)
        self.num_features=sum(features_sizes)
        self.loss_type=loss_type
        self.deep_layers=deep_layers
        self.activation=activation
        self.k=k #embedding size K
        self.use_LR=use_LR
        self.use_FM=use_FM
        self.use_MLP=use_MLP
        self.FM_ignore_interaction=[] if FM_ignore_interaction==None else FM_ignore_interaction
        self.attention_FM=attention_FM#同时代表attention hidden layer size
        self.use_NFM=use_NFM
        assert isinstance(self.FM_ignore_interaction,list),"FM_ignore_interaction type error"

        initializer=tf.contrib.layers.xavier_initializer()
        if self.use_LR:
            self.w=tf.Variable(initializer([self.num_features,1]))
            self.b=tf.Variable(initializer([1]))
        if self.use_FM or self.use_MLP:
            #[Embedding]
            self.embedding_weights=tf.Variable(initializer([self.num_features,k])) # sum_features_sizes,k
        if self.use_FM and self.attention_FM:
            self.attention_t=self.attention_FM # define t (type int)
            self.AFM_weights = {}
            self.AFM_weights['attention_W']=tf.Variable(initializer([self.k,self.attention_t]))#shape=(k,t)
            self.AFM_weights['attention_b']=tf.Variable(initializer([self.attention_t]))  # shape=(k,t)
            self.AFM_weights['projection_h']=tf.Variable(initializer([self.attention_t,1]))
            self.AFM_weights['projection_p']=tf.Variable(initializer([self.k,1]))
        if self.use_FM and self.use_NFM:
            self.NFM_weights={}
            self.NFM_weights['W1']=tf.Variable(initializer([self.k,self.k]))
            self.NFM_weights['W2']=tf.Variable(initializer([self.k,self.k]))
            self.NFM_weights['b1']=tf.Variable(initializer([self.k]))
            self.NFM_weights['b2']=tf.Variable(initializer([self.k]))
            self.NFM_weights['Wout']=tf.Variable(initializer([self.k,1]))
            self.NFM_weights['bout']=tf.Variable(initializer([1]))


        if self.use_MLP:
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
        print("Model Inited. Prediction Layer: LR %s; FM %s; MLP %s" % (use_LR,use_FM,use_MLP))

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
        for i,_ in enumerate(self.deep_layers):
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

    #FM_DependencyEliminate,求每个不同field交叉内积的和，但去掉了依赖的包含交叉
    def FMDE(self,embedding):#embedding:(None,field,k)
        cross_term=0
        for i in range(self.fields):
            for j in range(i+1,self.fields):
                #embedding[:,i,:] shape是(None,k)
                if (i,j) in self.FM_ignore_interaction:
                    continue
                cross_term=cross_term+tf.reduce_sum(embedding[:,i,:]*embedding[:,j,:],axis=1,keepdims=True)#(None,k)->(None,1)
        return cross_term

    def AFM(self,embedding,AFM_weights):
        cross_term=[]
        for i in range(self.fields):
            for j in range(i+1,self.fields):
                #embedding[:,i,:] shape是(None,k)
                if (i,j) in self.FM_ignore_interaction:
                    continue
                cross_term.append(embedding[:,i,:]*embedding[:,j,:])#(None,k) ！！不压缩成(None,1)
        cross_term=tf.stack(cross_term,axis=1)#(None,c,k)  c=cross term num. tf.stacke add new dim@1
        attention_space=tf.nn.relu(tf.tensordot(cross_term,AFM_weights['attention_W'],axes=[[2],[0]])+AFM_weights['attention_b']) #attention_W:(k,t) out:(None,c,t)  +attention_b(t)=(None,c,t)
        att_score=tf.tensordot(attention_space,AFM_weights['projection_h'],axes=[[2],[0]])#(None,c,t)*(t,1)=(None,c,1)
        self.normalize_att_score=tf.nn.softmax(att_score)#(None,c,1)
        attention_out=cross_term*self.normalize_att_score#(None,c,k)*(None,c,1)=(None,c,k)
        attention_out=tf.reduce_sum(attention_out,axis=1)#Sum pooling on cross terms. Get (None,k)
        return tf.matmul(attention_out,AFM_weights['projection_p'])#(None,k)*(k,1)=(None,1)

    def NFM(self,embedding,NFM_weights):
        square_sum=tf.square(tf.reduce_sum(embedding,axis=1))#(None,k)
        sum_square=tf.reduce_sum(tf.square(embedding),axis=1)#(None,k)
        cross_term_vec=square_sum-sum_square#(None,k)
        h1=self.activation(tf.matmul(cross_term_vec,NFM_weights['W1'])+NFM_weights['b1'])
        h2=self.activation(tf.matmul(h1,NFM_weights['W2'])+NFM_weights['b2'])
        return tf.matmul(h2,NFM_weights['Wout'])+NFM_weights['bout']

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        #data preprocess:对ids的每个features，label encoder都要从上一个的末尾开始。函数输入时则保证每个都从0起.
        for i,column in enumerate(ids_train.columns):
            if i>=1:
                ids_train.loc[:,column]=ids_train[column]+sum(self.features_sizes[:i])
                ids_test.loc[:, column]=ids_test[column]+sum(self.features_sizes[:i])
        if self.attention_FM:#储存为classs变量并用在get_attention里获取attention
            self.ids_train,self.ids_test,self.y_train,self.y_test = ids_train,ids_test,y_train,y_test

        self.ids=tf.placeholder(tf.int32,[None,self.fields])
        self.y=tf.placeholder(tf.float32,[None,1])


        if self.use_FM or self.use_MLP:
            self.embedding=self.Embedding(self.ids,self.embedding_weights)#(None,fields,k)

        self.pred=0
        if self.use_LR:
            self.pred=self.LR(self.ids,self.w,self.b)

        #only one FM will be used.
        if self.use_NFM:
            print("use NFM")
            self.pred+=self.NFM(self.embedding,self.NFM_weights)
        elif self.use_FM and not self.attention_FM:
            print("use FM")
            if len(self.FM_ignore_interaction)==0:#if self.use_FM and self.FM_ignore_interaction==[]
                self.pred+= self.FM2(self.embedding)
            if len(self.FM_ignore_interaction)>0:
                self.pred+=self.FMDE(self.embedding)
        elif self.use_FM and self.attention_FM:
            print("use AFM")
            self.pred+= self.AFM(self.embedding,self.AFM_weights)

        if self.use_MLP:
            MLP_in = tf.reshape(self.embedding, [-1, self.fields * self.k])
            self.pred+=self.MLP(MLP_in, self.weights, self.bias)
        assert self.pred is not None,"must have one predicion layer"



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

            test_loss=0.
            for bx,by in batcher(ids_test,y_test,batch_size):
                test_loss+=self.sess.run(self.loss,feed_dict={self.ids:bx,self.y:by})
            test_loss/=int(ids_test.shape[0]/batch_size)

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
                best_score = cur_min_loss #self.sess.run(self.loss, feed_dict={self.ids: ids_test, self.y: y_test, })
                print("Best Score:",best_score,' at round ',cur_best_rounds)
                return best_score
        self.sess.run(tf.tuple([tf.assign(var, best_weights[var.name]) for var in tf.trainable_variables()]))
        best_score=cur_min_loss #self.sess.run(self.loss, feed_dict={self.ids: ids_test, self.y: y_test,})
        print("Best Score:", best_score,' at round ',cur_best_rounds)
        return best_score


    def predict(self,ids_pred):
        for i,column in enumerate(ids_pred.columns):
            if i>=1:
                ids_pred.loc[:,column]=ids_pred[column]+sum(self.features_sizes[:i])
        self.output=self.sess.run(self.pred,feed_dict={self.ids: ids_pred,})
        return self.output

    def get_attention_mask(self):
        if not self.attention_FM:
            return
        self.attention_masks=[]
        for bx, by in batcher(self.ids_test,self.y_test, 500):
            self.attention_masks.append(self.sess.run(self.normalize_att_score, feed_dict={self.ids: bx, self.y: by}))
        return np.array(self.attention_masks)

if __name__ == '__main__':
    import pandas as pd
    np.random.seed(2019)
    data_dir="../data/movie_lens_100k/"
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