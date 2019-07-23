import tensorflow as tf
import numpy as np
import time
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from utils import batcher,isBetter,_build_regression_signature,_build_classification_signature
from tensorflow import python as tfpy
import os

class Alita_DeepFM(BaseEstimator):
    # features_sizes: array. number of features in every fields.e.g.[943,1682] user_nunique,movie_nunique
    def __init__(self,features_sizes,dense_features_size=0,loss_type='rmse',k=10,deep_layers=(256,256),activation=tf.nn.relu,use_LR=True,use_MLR=False,use_FM=True,use_MLP=True,FM_ignore_interaction=None,attention_FM=0,MLR_m=4,use_NFM=False,use_BiFM=False,use_SE=False,use_FiBiNet=False,use_AutoInt=False,autoint_params=None,dropout_keeprate=1.0,lambda_l2=0.0,hash_size=None,metric_type=None):
        self.features_sizes=features_sizes
        self.fields=len(features_sizes)
        self.num_features=sum(features_sizes) if hash_size is None else hash_size
        self.hash_size=hash_size
        self.dense_features_size=dense_features_size
        self.loss_type=loss_type
        self.metric_type=metric_type#only support AUC
        self.deep_layers=deep_layers
        self.activation=activation
        self.k=k #embedding size K
        self.use_LR=use_LR
        self.use_MLR = use_MLR
        self.MLR_m = MLR_m
        self.use_FM=use_FM
        self.use_MLP=use_MLP
        self.FM_ignore_interaction=[] if FM_ignore_interaction==None else FM_ignore_interaction
        self.attention_FM=attention_FM#同时代表attention hidden layer size
        self.use_NFM=use_NFM
        self.use_BiFM=use_BiFM
        self.use_SE=use_SE
        self.use_FiBiNet=use_FiBiNet
        self.use_AutoInt=use_AutoInt
        self.autoint_params=autoint_params if autoint_params is not None else {"autoint_d":8,'autoint_heads':2,"autoint_layers":3,'relu':True,'use_res':True}

        self.coldStartAvg=False
        self.dropout_keeprate=dropout_keeprate
        self.lambda_l2=lambda_l2

        self.c=0#cross terms
        for i in range(self.fields):
            for j in range(i+1,self.fields):
                if (i,j) in self.FM_ignore_interaction:
                    continue
                self.c+=1

        assert isinstance(self.FM_ignore_interaction,list),"FM_ignore_interaction type error"

        initializer=tf.contrib.layers.xavier_initializer()
        if self.use_LR:
            self.w=tf.Variable(initializer([self.num_features,1]))
            self.b=tf.Variable(initializer([1]))
        if self.use_MLR:
            self.MLR_u=tf.Variable(initializer([self.num_features,self.MLR_m]))
            self.MLR_w=tf.Variable(initializer([self.num_features,self.MLR_m]))

        if self.use_FM or self.use_MLP or self.use_AutoInt:
            #[Embedding]
            self.embedding_weights=tf.Variable(initializer([self.num_features,k])) # sum_features_sizes,k
        if self.use_FM and self.attention_FM:
            self.attention_t=self.attention_FM # define t (type int)
            self.AFM_weights = {}
            self.AFM_weights['attention_W']=tf.Variable(initializer([self.k,self.attention_t]))#shape=(k,t)
            self.AFM_weights['attention_b']=tf.Variable(initializer([self.attention_t]))  # shape=(k,t)
            self.AFM_weights['projection_h']=tf.Variable(initializer([self.attention_t,1]))
            self.AFM_weights['projection_p'] = tf.Variable(initializer([self.k, 1]))

            #SAFM:
            #self.AFM_weights['downsample_W'] = tf.Variable(initializer([self.c, 1]))

            #CFM:
            self.AFM_weights['conv_W']=tf.Variable(initializer([self.c*self.k,1]))
            self.oup=1
            #todo
            self.AFM_weights['filter'] = tf.Variable(initializer([self.c, self.k, 1 ,self.oup]))#4D [filter_height, filter_width, in_channels, out_channels]
            #self.AFM_weights['filter'] = tf.Variable(tf.ones([self.c, self.k, 1 ,self.oup]))#4D [filter_height, filter_width, in_channels, out_channels]
            self.AFM_weights['proj']=tf.Variable(initializer([self.oup,1]))

            #CNFM:
            h1,h2=8,4
            self.h1,self.h2=h1,h2
            self.AFM_weights['h1_W']=tf.Variable(initializer([self.c,h1]))
            self.AFM_weights['h1_b']=tf.Variable(initializer([h1]))
            self.AFM_weights['h2_W']=tf.Variable(initializer([h1,h2]))
            self.AFM_weights['h2_b']=tf.Variable(initializer([h2]))
            self.AFM_weights['out_W']=tf.Variable(initializer([h2,1]))
            self.AFM_weights['out_b']=tf.Variable(initializer([1]))
            self.AFM_weights['filter_h1']= tf.Variable(initializer([h1,self.k,1,1]))
            self.AFM_weights['filter_h2']= tf.Variable(initializer([h2,self.k,1,1]))
            self.AFM_weights['filter_out']=tf.Variable(initializer([1, self.k,1,1]))


        if self.use_FM and self.use_NFM:
            self.NFM_weights={}
            self.NFM_weights['W1']=tf.Variable(initializer([self.k,self.k]))
            self.NFM_weights['W2']=tf.Variable(initializer([self.k,self.k]))
            self.NFM_weights['b1']=tf.Variable(initializer([self.k]))
            self.NFM_weights['b2']=tf.Variable(initializer([self.k]))
            self.NFM_weights['Wout']=tf.Variable(initializer([self.k,1]))
            #todo NFM simple k->1's Wout init (+0.4%)
            #self.NFM_weights['Wout'] = tf.Variable(tf.ones([self.k,1]))
            self.NFM_weights['bout']=tf.Variable(initializer([1]))

        if self.use_BiFM:
            self.bilinear_weights = {}
            self.bilinear_weights['w'] = tf.Variable(initializer([self.k, self.k]))
            self.bilinear_weights['wse'] = tf.Variable(initializer([self.k, self.k]))
        if self.use_SE:
            self.SE_weights={}
            self.SE_weights['W1']=tf.Variable(initializer([self.fields,self.fields//2]))
            self.SE_weights['W2'] = tf.Variable(initializer([self.fields//2, self.fields]))
        if self.use_FiBiNet:
            self.FiBiNet_weights={}
            self.FiBiNet_weights['W1']=tf.Variable(initializer([self.c*self.k*2,(self.c*self.k)//20]))
            self.FiBiNet_weights['b1'] = tf.Variable(initializer([(self.c*self.k)//20]))
            self.FiBiNet_weights['W2']=tf.Variable(initializer([(self.c*self.k)//20,1]))
            self.FiBiNet_weights['b2'] = tf.Variable(initializer([1]))

        if self.use_AutoInt:
            #d=8 head=2 layer=3
            self.autoint_d=    self.autoint_params['autoint_d']
            self.autoint_head= self.autoint_params['autoint_heads']
            self.autoint_layer=self.autoint_params['autoint_layers']
            print("AutoInt params:",self.autoint_params)
            self.AutoInt_weights={}
            for layer in range(self.autoint_layer):
                first_dim = self.k if layer==0 else self.autoint_d*self.autoint_head
                self.AutoInt_weights['W_query_'+str(layer+1)]=tf.Variable(initializer([first_dim,self.autoint_d*self.autoint_head]))
                self.AutoInt_weights['W_key_'+str(layer+1)]=tf.Variable(initializer([first_dim,self.autoint_d*self.autoint_head]))
                self.AutoInt_weights['W_value_'+str(layer+1)]=tf.Variable(initializer([first_dim,self.autoint_d*self.autoint_head]))
                self.AutoInt_weights['W_res_'+str(layer+1)] = tf.Variable(initializer([first_dim, self.autoint_d*self.autoint_head]))

            self.AutoInt_weights['W_out'] = tf.Variable(initializer([self.fields*self.autoint_d*self.autoint_head,1]))
            self.AutoInt_weights['b_out'] = tf.Variable(initializer([1]))

        if self.use_MLP:
            #MLP weights
            self.weights, self.bias = {}, {}
            self.MLP_n_input=k*self.fields+self.dense_features_size
            for i,layer in enumerate(self.deep_layers):
                if i==0:
                    self.weights['h'+str(i+1)]=tf.Variable(initializer([self.MLP_n_input,layer]))
                else:
                    self.weights['h'+str(i+1)] = tf.Variable(initializer([self.deep_layers[i-1], layer]))
                self.bias['b'+str(i+1)]=tf.Variable(initializer([layer]))
            self.weights['out']= tf.Variable(initializer([self.deep_layers[-1],1]))
            self.bias['out'] = tf.Variable(initializer([1]))
        print("[Info] Model Inited. Prediction Layer: LR %s; FM %s; MLP %s; AutoInt %s" % (use_LR,use_FM,use_MLP,use_AutoInt))

    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.7
        return tf.Session(config=config)


    def build_model(self,export_path_base='C:/Users/13414/tfserving/tme_ad/saved/classification',version=1):
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(version)))
        if not os.path.isdir(export_path):
            print("making dir:",export_path)
            os.mkdir(export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        predict_signature_inputs={"x": tf.saved_model.utils.build_tensor_info(self.ids)}
        predict_signature_outputs = {"y": tf.saved_model.utils.build_tensor_info(self.pred)}
        predict_signature_def=(
        tf.saved_model.signature_def_utils.build_signature_def(
            predict_signature_inputs, predict_signature_outputs,
            tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        #import from utils:
        target_signature_func=_build_regression_signature if self.loss_type in ['mse','rmse'] else _build_classification_signature
        signature_def_map = {
            "target":
                target_signature_func(self.ids, self.pred),
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                predict_signature_def
        }
        builder.add_meta_graph_and_variables(
            self.sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_def_map,
            assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
        )
        builder.save()


    #todo bug: 要keepdims 输出(None,1)而不是(None,)
    def LR(self,ids,w,b):
        #ids:(None,field)  w:(num_features,1)  out:(None,field,1) ->reshape(None,fields)
        return tf.reduce_sum(tf.reshape(tf.nn.embedding_lookup(w,ids),[-1,self.fields]),axis=1,keepdims=True)+b #(N,1)

    # TODO problem of sigmoid. learner分数不能在加sigmoid了，因为最终loss里有.
    def MLR(self,ids,u,w):
        region_scores= tf.nn.softmax(tf.reduce_sum(tf.nn.embedding_lookup(u,ids),axis=1))#N,f,m (m=MLR_m aka pieces) ---reshape---->None,m
        learner_scores= tf.reduce_sum(tf.nn.embedding_lookup(w,ids),axis=1)#N,f,m->None,m
        return tf.reduce_sum(tf.multiply(region_scores,learner_scores),axis=1,keepdims=True)#N,1

    def Embedding(self,ids,params):
        #params:self.embedding(sum_features_sizes,k)   ids:(None,fields)  out=shape(ids)+shape(params[1:])=(None,fields,k)
        return tf.nn.embedding_lookup(params,ids),tf.nn.l2_loss(params)

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

    def Bilinear_FM(self,embedding,bilinear_weights,se_emb):#embedding:(None,field,k) bilinear_weights:(N,k,k)
        b_weights=bilinear_weights['w'] if se_emb else bilinear_weights['wse']
        cross_term=[]
        for i in range(self.fields):
            for j in range(i+1,self.fields):
                #embedding[:,i,:] shape是(None,k)
                cross_term.append(tf.matmul(embedding[:,i,:],b_weights)*embedding[:,j,:])#(None,k)
                #cross_term.append(embedding[:,i,:]*embedding[:,j,:])#(None,k) ！！不压缩成(None,1)
        cross_term=tf.stack(cross_term,axis=1)#(None,c,k)
        return cross_term
        #return tf.expand_dims(tf.reduce_sum(cross_term,axis=[1,2]),axis=1)#(None,1)

        #fc=tf.layers.Dense(24,activation=tf.nn.relu,kernel_initializer=tfpy.keras.initializers.glorot_normal())(tf.reshape(cross_term,[-1,cross_term.shape[1]*cross_term.shape[2]]))
        #fc=tf.layers.Dense(1,activation=tf.nn.relu,kernel_initializer=tfpy.keras.initializers.glorot_normal())(fc)
        #print("FC.shape",fc.shape)
        #return fc

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
        attention_out=tf.nn.dropout(attention_out, keep_prob=self.dropout_keeprate)
        return tf.matmul(attention_out,AFM_weights['projection_p']),tf.nn.l2_loss(AFM_weights['attention_W'])#(None,k)*(k,1)=(None,1)
        #with l2 reg

    def NFM(self,embedding,NFM_weights):
        square_sum=tf.square(tf.reduce_sum(embedding,axis=1))#(None,k)
        sum_square=tf.reduce_sum(tf.square(embedding),axis=1)#(None,k)
        cross_term_vec=square_sum-sum_square#(None,k)
        h1=self.activation(tf.matmul(cross_term_vec,NFM_weights['W1'])+NFM_weights['b1'])
        #h2=self.activation(tf.matmul(h1,NFM_weights['W2'])+NFM_weights['b2'])

        #todo simple NFM use cross_term vec, else use h1
        #return tf.matmul(cross_term_vec,NFM_weights['Wout'])+NFM_weights['bout']
        return tf.matmul(h1, NFM_weights['Wout']) + NFM_weights['bout']


    #直接用矩阵乘法把(None,c,k)->(None,k)->(None,1)
    def SAFM(self,embedding,AFM_weights):
        cross_term=[]
        for i in range(self.fields):
            for j in range(i+1,self.fields):
                #embedding[:,i,:] shape是(None,k)
                if (i,j) in self.FM_ignore_interaction:
                    continue
                cross_term.append(embedding[:,i,:]*embedding[:,j,:])#(None,k) ！！不压缩成(None,1)
        cross_term=tf.stack(cross_term,axis=1)#(None,c,k)  c=cross term num. tf.stacke add new dim@1
        #print("AFM_weights['downsample_W']",AFM_weights['downsample_W'].shape)
        out=tf.tensordot(cross_term,AFM_weights['downsample_W'],axes=[[1],[0]])#(None,c,k)*(c,1)=(None,k,1)
        out=tf.reduce_sum(out,axis=2)#Dim Reduce@2. Get (None,k)
        return tf.matmul(out,AFM_weights['projection_p'])#(None,k)*(k,1)=(None,1)

    #convolution based FM  (None,c,k)->(None,c*k)->(None,1)
    def CFM(self,embedding,AFM_weights):
        cross_term=[]
        for i in range(self.fields):
            for j in range(i+1,self.fields):
                #embedding[:,i,:] shape是(None,k)
                if (i,j) in self.FM_ignore_interaction:
                    continue
                cross_term.append(embedding[:,i,:]*embedding[:,j,:])#(None,k) ！！不压缩成(None,1)
        cross_term=tf.stack(cross_term,axis=1)#(None,c,k)  c=cross term num. tf.stacke add new dim@1
        cross_term = tf.nn.dropout(cross_term, keep_prob=self.dropout_keeprate)
        #imp1 matmul style
        #cross_term=tf.reshape(cross_term,shape=[-1,self.c*self.k])
        #out=tf.matmul(cross_term,AFM_weights['conv_W'])#(None,ck)*(ck,1)=(None,1)
        #return out,tf.nn.l2_loss(AFM_weights['conv_W'])

        #imp2 conv2d style: tune self.oup=1 is best.
        cross_term = tf.expand_dims(cross_term,axis=-1)
        out=tf.nn.conv2d(cross_term,AFM_weights['filter'],strides=[1,1,1,1],padding='VALID')#N,1,1,1 iff oup=1; #N,1,1,k iff oup=self.oup
        out=tf.reshape(out,shape=[-1,self.oup])#(N,1,1,oup)->(None,oup)
        return (out,tf.nn.l2_loss(AFM_weights['filter']) ) if self.oup==1 else (tf.matmul(out,AFM_weights['proj']),tf.nn.l2_loss(AFM_weights['filter']))

    def CNFM(self, embedding, AFM_weights):
        cross_term=[]
        for i in range(self.fields):
            for j in range(i+1,self.fields):
                #embedding[:,i,:] shape是(None,k)
                if (i,j) in self.FM_ignore_interaction:
                    continue
                cross_term.append(embedding[:,i,:]*embedding[:,j,:])#(None,k) ！不压缩成(None,1)

        cross_term=tf.stack(cross_term,axis=1)#(None,c,k)  c=cross term num
        out0=tf.nn.conv2d(tf.expand_dims(cross_term, axis=-1),AFM_weights['filter'],strides=[1,1,1,1],padding='VALID')#N,1,1,1 iff oup=1; #N,1,1,k iff oup=self.oup
        #out0=tf.matmul(tf.reshape(out0,shape=[-1,self.oup]),AFM_weights['proj'])#(N,1,1,oup)->(None,oup) 再*(oup,1) = None,1
        out0=tf.reshape(out0,shape=[-1,1])

        h1=tf.transpose(tf.nn.relu(tf.tensordot(cross_term,AFM_weights['h1_W'],axes=[[1],[0]])+AFM_weights['h1_b']),perm=[0,2,1]) #(N,c,k)*(c,h1)=(N,k,h1)  -> (N,h1,k)
        out1 = tf.nn.conv2d(tf.expand_dims(h1, axis=-1), AFM_weights['filter_h1'], strides=[1, 1, 1, 1], padding='VALID')
        out1=tf.reshape(out1,shape=[-1,1])

        h2=tf.transpose(tf.nn.relu(tf.tensordot(h1,AFM_weights['h2_W'],axes=[[1],[0]])+AFM_weights['h2_b']),perm=[0,2,1]) #(N,h1,k)*(h1,h2)=(N,k,h2) transpose N,h2,k
        out2 = tf.nn.conv2d(tf.expand_dims(h2, axis=-1), AFM_weights['filter_h2'], strides=[1, 1, 1, 1], padding='VALID')
        out2 = tf.reshape(out2, shape=[-1, 1])

        h3=tf.transpose(tf.nn.relu(tf.tensordot(h2,AFM_weights['out_W'],axes=[[1],[0]])+AFM_weights['out_b']),perm=[0,2,1]) #(N,h2,k)*(h2,1)=(N,k,1) transpose N,1,k
        out3 = tf.nn.conv2d(tf.expand_dims(h3, axis=-1), AFM_weights['filter_out'], strides=[1, 1, 1, 1], padding='VALID')
        out3 = tf.reshape(out3, shape=[-1, 1])
        return out0+out1+out2+out3 #(None,1)

    #Embedding:(none,f,k)
    def AutoInt(self,embedding,AutoInt_Weights,layer):
        querys=tf.tensordot(embedding,AutoInt_Weights['W_query_'+str(layer+1)],axes=[[2],[0]])#(N,f,k) * (k,d*head)= N,f,d*head
        keys=  tf.tensordot(embedding,AutoInt_Weights['W_key_'+str(layer+1)],axes=[[2],[0]])#N,f,d*head
        values = tf.tensordot(embedding, AutoInt_Weights['W_value_'+str(layer+1)], axes=[[2], [0]])#N,f,d*head

        #head,N,f,d
        querys = tf.stack(tf.split(querys, self.autoint_head, axis=2))#default axis=0 (head,N,f,d)
        keys = tf.stack(tf.split(keys, self.autoint_head, axis=2))
        values = tf.stack(tf.split(values, self.autoint_head, axis=2))

        self.normalize_autoint_att_score=tf.nn.softmax(tf.matmul(querys,keys,transpose_b=True))#heads,N,f,f
        out=tf.matmul(self.normalize_autoint_att_score,values)#heads,N,f,d
        out = tf.concat(tf.split(out, self.autoint_head,axis=0), axis=-1) #[(N,f,d)*heads个] --concat--> [1,N,f,d*heads]
        out = tf.squeeze(out, axis=0)  # (N,f,d*heads)

        if self.autoint_params['use_res']:
            #                 first round：N,f,k    k,d*heads  | others:  N,f,d*heads  d*heads,d*heads
            out+=tf.tensordot(embedding,AutoInt_Weights['W_res_'+str(layer+1)],axes=[[2],[0]])

        if self.autoint_params['relu']:
            out=tf.nn.relu(out)
        return out


    def fit(self,ids_train,ids_test,y_train,y_test,dense_train=None,dense_test=None,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20,):
        start_time = time.time()
        #[bug fix]mutable prevention 19/06/27
        ids_train=ids_train.copy()
        ids_test=ids_test.copy()

        self.batch_size=batch_size
        #data preprocess:对ids的每个features，label encoder都要从上一个的末尾开始。函数输入时则保证每个都从0起.
        if self.hash_size is None:
            for i,column in enumerate(ids_train.columns):
                if i>=1:
                    ids_train.loc[:,column]=ids_train[column]+sum(self.features_sizes[:i])
                    ids_test.loc[:, column]=ids_test[column]+sum(self.features_sizes[:i])
        if self.attention_FM or self.use_AutoInt:#储存为classs变量并用在get_attention里获取attention
            self.ids_train,self.ids_test,self.y_train,self.y_test = ids_train,ids_test,y_train,y_test

        self.ids=tf.placeholder(tf.int32,[None,self.fields])
        self.dense_inputs=tf.placeholder(tf.float32,[None,self.dense_features_size])
        self.y=tf.placeholder(tf.float32,[None,1])
        self.L2_reg = 0

        self.dropout_keeprate_holder=tf.placeholder(tf.float32)

        embed_L2=0
        if self.use_FM or self.use_MLP or self.use_AutoInt:
            self.embedding,embed_L2=self.Embedding(self.ids,self.embedding_weights)#(None,fields,k)

        if self.use_SE:
            self.embeddingSE=self.SELayer(self.embedding,self.SE_weights)

        self.pred=0
        if self.use_LR:
            #bug detected. LR didn't keepdims
            self.pred=self.LR(self.ids,self.w,self.b)
        if self.use_MLR:
            print("use Mix of LR.")
            self.pred+=self.MLR(self.ids,self.MLR_u,self.MLR_w)

        #only one FM will be used.
        if self.use_NFM:
            print("use NFM")
            self.pred+=self.NFM(self.embedding,self.NFM_weights)
        elif self.use_BiFM:
            if self.use_SE:
                cross_term = tf.concat([self.Bilinear_FM(self.embedding, self.bilinear_weights, se_emb=False),
                                        self.Bilinear_FM(self.embeddingSE, self.bilinear_weights, se_emb=True),
                                        ],axis=-1)  # N,c,2k
                if self.use_FiBiNet:
                    print("use FiBiNet")#deep backend
                    cross_term=tf.reshape(cross_term,[-1,self.c*self.k*2])#None,2ck
                    cross_term=tf.nn.relu(tf.matmul(cross_term,self.FiBiNet_weights['W1'])+self.FiBiNet_weights['b1'])
                    self.pred += (tf.matmul(cross_term,self.FiBiNet_weights['W2'])+self.FiBiNet_weights['b2'])
                else:
                    print("use Fibifm")
                    self.pred += tf.expand_dims(tf.reduce_sum(cross_term, axis=[1, 2]), axis=1)  # N,1

            else:
                print("use bifm")
                cross_term = self.Bilinear_FM(self.embedding, self.bilinear_weights,se_emb=False)  # N,c,k
                self.pred += tf.expand_dims(tf.reduce_sum(cross_term, axis=[1, 2]), axis=1)  # N,1

        elif self.use_FM and not self.attention_FM:
            print("use FM")
            if len(self.FM_ignore_interaction)==0:#if self.use_FM and self.FM_ignore_interaction==[]
                self.pred += self.FM2(self.embedding)
            if len(self.FM_ignore_interaction)>0:
                self.pred+=self.FMDE(self.embedding)
        elif self.use_FM and self.attention_FM:
            print("use CFM")
            #afm_out,reg= self.AFM(self.embedding,self.AFM_weights)
            afm_out,reg= self.CFM(self.embedding,self.AFM_weights)
            self.pred+=afm_out
            self.L2_reg+=reg

        if self.use_AutoInt:
            self.y_deep=self.embedding
            for _l in range(self.autoint_params['autoint_layers']):
                self.y_deep=self.AutoInt(self.y_deep,self.AutoInt_weights,layer=_l)#N,f,d
            self.pred+=tf.matmul(tf.reshape(self.y_deep,shape=[-1,self.fields*self.autoint_d*self.autoint_head]),self.AutoInt_weights['W_out'])+self.AutoInt_weights['b_out']
        if self.use_MLP:
            MLP_in = tf.reshape(self.embedding, [-1, self.fields * self.k])#(N,f*k)
            if self.dense_features_size>0:
                MLP_in=tf.concat([MLP_in,self.dense_inputs],axis=1)#(N,f*k+dense)
            self.pred+=self.MLP(MLP_in, self.weights, self.bias)
            #self.pred=self.SqueezeEmbLR(self.embedding,self.SqueezeEmb_LRWeight)
        assert self.pred is not None,"must have one predicion layer"


        if self.loss_type=='rmse':
            self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.pred)))
        elif self.loss_type=='mse':
            self.loss = tf.reduce_mean(tf.square(self.y-self.pred))
        elif self.loss_type in ['binary_crossentropy','binary','logloss']:
            self.loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,logits=self.pred))
        else:
            raise Exception("Loss type %s not supported"%self.loss_type)

        #todo EMBEDL2 coef
        self.loss += self.lambda_l2*self.L2_reg #+ embed_L2*1e-5
        self.optimizer=tf.train.AdamOptimizer(lr).minimize(self.loss)

        if self.metric_type is not None:
            assert self.metric_type=='auc'
            assert self.loss_type in ['binary_crossentropy', 'binary', 'logloss']
            #tf.auc mode: remove sklearn auc part
            #self.loss=tf.metrics.auc(labels=self.y,predictions=tf.nn.sigmoid(self.pred))

        self.sess=self._init_session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        cur_best_rounds=0

        is_greater_better = False if self.metric_type is None else True #默认Loss越小越好
        cur_min_loss = 1e8 if not is_greater_better else -1e8
        best_weights = {v.name: v.eval(self.sess) for v in tf.trainable_variables()}

        for epoch in range(N_EPOCH):
            train_loss=0.;y_preds_train=[]
            total_batches=int(ids_train.shape[0]/batch_size)
            # id input + dense input
            for bx,bx_dense,by in batcher(ids_train,y_train,X_dense=dense_train,batch_size=batch_size,hash_size=self.hash_size):
                if self.dense_features_size>0:
                    _,l=self.sess.run([self.optimizer,self.loss],feed_dict={self.ids:bx,self.y:by,self.dense_inputs:bx_dense,self.dropout_keeprate_holder:self.dropout_keeprate})
                else:
                    _,l=self.sess.run([self.optimizer,self.loss],feed_dict={self.ids:bx,self.y:by,self.dropout_keeprate_holder:self.dropout_keeprate})
                train_loss+=l #if not self.metric_type else l[1]
                if self.metric_type:
                    y_preds_train.append(self.sess.run(self.pred,feed_dict={self.ids:bx,self.dense_inputs:bx_dense,self.dropout_keeprate_holder:1.0})) \
                    if self.dense_features_size>0 \
                    else y_preds_train.append(self.sess.run(self.pred,feed_dict={self.ids:bx,self.dropout_keeprate_holder:1.0}))
            train_loss/=total_batches

            if self.coldStartAvg:
                print("Cold Start Averaging start") if epoch==0 else None
                self.coldStartAvgTool()

            #todo movielens afm rounded
            test_loss=0.;y_preds=[]
            for bx,bx_dense,by in batcher(ids_test,y_test,X_dense=dense_test,batch_size=batch_size,hash_size=self.hash_size):
                if self.dense_features_size > 0:
                    l=self.sess.run(self.loss,feed_dict={self.ids:bx,self.y:by,self.dense_inputs:bx_dense,self.dropout_keeprate_holder:1.0})
                else:
                    l=self.sess.run(self.loss,feed_dict={self.ids:bx,self.y:by,self.dropout_keeprate_holder:1.0})
                test_loss+=l #if not self.metric_type else l[1]
                if self.metric_type:
                    y_preds.append(self.sess.run(self.pred,feed_dict={self.ids:bx,self.dense_inputs:bx_dense,self.dropout_keeprate_holder:1.0})) \
                    if self.dense_features_size>0 \
                    else y_preds.append(self.sess.run(self.pred,feed_dict={self.ids:bx,self.dropout_keeprate_holder:1.0}))


            test_loss/=int(ids_test.shape[0]/batch_size)
            '''
            y_pred=np.concatenate(y_preds, axis=0).reshape((-1))
            predictions_bounded = np.maximum(y_pred, np.ones(len(y_pred)) * -1)  # bound the lower values
            predictions_bounded = np.minimum(predictions_bounded, np.ones(len(y_pred)) * 1)  # bound the higher values
            # override test_loss
            test_loss = np.sqrt(np.mean(np.square(y_test.reshape(predictions_bounded.shape)- predictions_bounded)))
            '''
            #sklearn auc mode
            if self.metric_type:# override test_loss
                self.y_pred_train=np.concatenate(y_preds_train,axis=0)
                self.y_pred = np.concatenate(y_preds, axis=0)
                train_loss=roc_auc_score(y_train,self.y_pred_train)
                test_loss = roc_auc_score(y_test,self.y_pred)

            metrics_='loss' if self.metric_type is None else 'auc'
            print("epoch:%s train_%s:%s test_%s:%s" %(epoch+1,metrics_,train_loss,metrics_,test_loss))
            #print("self.pred=",self.sess.run(self.pred,feed_dict={self.ids:ids_test,self.y:y_test}))
            #print("self.y=",y_test)

            if isBetter(test_loss,cur_min_loss,is_greater_better):
                cur_min_loss=test_loss
                cur_best_rounds=epoch+1
                best_weights = {v.name: v.eval(self.sess) for v in tf.trainable_variables()}
            if epoch+1-cur_best_rounds>=early_stopping_rounds:
                print("[Early Stop]Early Stopping because not improved for %s rounds" % early_stopping_rounds)
                self.sess.run(tf.tuple([tf.assign(var, best_weights[var.name]) for var in tf.trainable_variables()]))
                best_score = cur_min_loss #self.sess.run(self.loss, feed_dict={self.ids: ids_test, self.y: y_test, })
                print("[Early Stop]Best Score:",best_score,' at round ',cur_best_rounds)
                print("Train finish. Fit time:%.2f seconds. Epoch time:%.2f seconds"%(time.time()-start_time,(time.time()-start_time)/(epoch+1)))
                return best_score

            #auc reset op
            self.sess.run(tf.local_variables_initializer())

        self.sess.run(tf.tuple([tf.assign(var, best_weights[var.name]) for var in tf.trainable_variables()]))
        best_score=cur_min_loss #self.sess.run(self.loss, feed_dict={self.ids: ids_test, self.y: y_test,})
        print("[Epoch Maxi]Best Score:", best_score,' at round ',cur_best_rounds)
        print("Train finish. Fit time:%.2f seconds. Epoch time:%.2f seconds"%(time.time()-start_time,(time.time()-start_time)/N_EPOCH))
        return best_score


    def predict(self,ids_pred,dense_pred=None):
        # [bug fix]mutable prevention 19/06/27
        start_time=time.time()
        ids_pred=ids_pred.copy()
        if self.hash_size is None:
            for i,column in enumerate(ids_pred.columns):
                if i>=1:
                    ids_pred.loc[:,column]=ids_pred[column]+sum(self.features_sizes[:i])
        outputs = []
        #self.ids_pred=ids_pred 0708 del
        for bx,bx_dense in batcher(ids_pred,y_=None,X_dense=dense_pred,batch_size=self.batch_size,hash_size=self.hash_size): #y=None
            if self.dense_features_size > 0:
                outputs.append(self.sess.run(self.pred, feed_dict={self.ids: bx,self.dense_inputs:bx_dense,self.dropout_keeprate_holder:1.0}))
            else:#bx_dense==None
                outputs.append(self.sess.run(self.pred, feed_dict={self.ids: bx,self.dropout_keeprate_holder:1.0}))
        self.output=np.concatenate(outputs, axis=0)   #.reshape((-1))
        time_cost=time.time()-start_time
        print("Time cost: %3f seconds, average epoch time:%2f ms" % (time_cost,(time_cost/len(outputs))*1000))
        return self.output

    def SELayer(self,embedding,SE_weights):
        squeeze=tf.reduce_mean(embedding,axis=2) #N,f
        squeeze=tf.nn.relu(tf.matmul(squeeze,SE_weights['W1'])) #squeeze:N,f W1:f,f//2 -> N,f//2
        excit  =tf.nn.sigmoid(tf.matmul(squeeze,SE_weights['W2']))#N,f
        excit = tf.expand_dims(excit,axis=2)#N,f,1
        return tf.multiply(embedding,excit)#

    def __del__(self):
        try:
            if self.sess is not None:
                self.sess.close()
        except:
            pass

    def get_attention_mask(self):
        if not self.attention_FM and not self.use_AutoInt:
            return
        self.attention_masks=[]
        for bx,bx_dense,by in batcher(X_=self.ids_test,y_=self.y_test, batch_size=500,hash_size=self.hash_size):#dense
            if self.attention_FM:
                self.attention_masks.append(self.sess.run(self.normalize_att_score, feed_dict={self.ids: bx, self.y: by,self.dropout_keeprate_holder:1.0}))
            else:
                self.attention_masks.append(self.sess.run(self.normalize_autoint_att_score, feed_dict={self.ids: bx, self.y: by,self.dropout_keeprate_holder:1.0}))
        #return np.array(self.attention_masks)
        return self.attention_masks

    def coldStartAvgTool(self):
        ops = []
        cold_start_idx=0
        for i,sizes in enumerate(self.features_sizes):
            if self.use_LR:
                #num_features,1
                op=self.w[cold_start_idx].assign(tf.reduce_mean(self.w[cold_start_idx+1:cold_start_idx+self.features_sizes[i]],axis=0,keepdims=False))
                ops.append(op)
            #EMBEDDING
            if self.use_FM or self.use_MLP or self.use_AutoInt:
                op=self.embedding_weights[cold_start_idx,:].assign(tf.reduce_mean(self.embedding_weights[cold_start_idx+1:cold_start_idx+self.features_sizes[i],:],axis=0,keepdims=False))
                ops.append(op)
            cold_start_idx=cold_start_idx+self.features_sizes[i]
        self.sess.run(ops)


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