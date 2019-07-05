
'''

    def MLP(self,x,weights,bias):
        last_layer=None
        for i,_ in enumerate(self.deep_layers):
            if i==0:
                last_layer=self.activation(tf.matmul(x,weights['h1'])+bias['b1'])
            else:
                this_layer=self.activation(tf.matmul(last_layer, weights['h'+str(i+1)]) + bias['b'+str(i+1)])
                last_layer=this_layer
        if(weights['out'].shape[0]==1):
            print("[Warning] Neglect last projection 1 to 1.")
            return last_layer
        else:
            return tf.matmul(last_layer,weights['out'])+bias['out']

    def addFM(self,embedding):
        embedding=tf.log(tf.nn.relu(embedding))
        cross_term=0
        for i in range(self.fields):
            for j in range(i+1,self.fields):
                #embedding[:,i,:] shape是(None,k)
                cross_term=cross_term+tf.reduce_sum(tf.exp(embedding[:,i,:]+embedding[:,j,:]),axis=1,keepdims=True)#(None,k)->(None,1)
        return cross_term

    def LogTransformEmbedding(self,ids,params):
        #params:self.embedding(sum_features_sizes,k)   ids:(None,fields)  out=shape(ids)+shape(params[1:])=(None,fields,k)
        return tf.log((e-1/e)*tf.nn.sigmoid(tf.nn.embedding_lookup(params,ids))+(1/e))


'''

'''
#190430

e=2.718281828459
        self.SqueezeEmb_LRWeight = {}
        self.SqueezeEmb_LRWeight['W'] = tf.Variable(initializer([self.fields, 1]))
        self.SqueezeEmb_LRWeight['W1']=tf.Variable(initializer([self.fields,3]))
        self.SqueezeEmb_LRWeight['W2'] = tf.Variable(initializer([3, 1]))
        self.SqueezeEmb_LRWeight['p'] = tf.Variable(initializer([self.k, 1]))

    def SqueezeEmbLR(self,embedding,weights):
        logTrnasform=True
        EXP=True
        if logTrnasform:
            embedding=tf.log((e - 1 / e) * tf.nn.sigmoid(embedding) + (1 / e))
            #embedding = tf.log((e-1)*tf.nn.sigmoid(embedding)+1)
        L=1
        if L==1:
            print('Testing')
            #L=1 #(None,f,k)*(f,1)->(None,k)->(None,1)  跟SAFM除了input space不同外都相同
            t=tf.reshape(tf.tensordot(embedding,weights['W'],axes=[[1],[0]]),shape=(-1,self.k))#None,k

            if EXP:
                t=tf.exp(t)
            #return tf.reduce_sum(t,axis=1)#(None,1) #~0.9  after log:~1.01 都不行
            return tf.matmul(t,weights['p']) # <0.6  k还是要带权重的好. 学习能力强

        else:
            #L=2 （None,f,k）->matmul->(None,k,f(t))->(None,k,1)->matmul->(None,1)
            t=tf.tensordot(embedding,weights['W1'],axes=[[1],[0]])#None,f,k*f,f =N,k,f
            t = tf.reshape(tf.tensordot(t, weights['W2'],axes=[[2],[0]]),shape=(-1,self.k))#Nkf * f1 =NK1-NK
            if EXP:
                t=tf.exp(t)
            return tf.matmul(t,weights['p'])
#performance
'''

'''
#190430
           TestNewIdea:
           MLP(features_sizes, deep_layers=(1,), k=256 fit params:lr=0.001,N_EPOCH=100,batch_size=4096,early_stopping_rounds=15
           original 不稳定: 0.8040 0.5927 [0.9426522484003089, 0.5992870099975481, 0.5927495460523411, 0.9426565680023961, 0.9426524416800905]
original+neglect 1-1 proj : 0.9821 0.9107 [0.9106921805962521, 1.0, 1.0, 1.0, 1.0]

            SqueezeEmbdLR : (None,f,k)-ruduceSum->(None,f)——matmul->(None,1)
            0.5964 0.5958 [0.5966447129634915, 0.5957894468950813, 0.5966280765507812, 0.5967074255750915, 0.5960330108343402]

         1 hidden No logTrans（None,f,k）->matmul->(None,k)->matmul->(None,1)
          0.5966 0.5946 [0.5945536414960693, 0.5968237385849661, 0.5980923460942627, 0.5963106192298754, 0.597288048995474]
         1 hidden LogTrans
          0.5923 0.5909 [0.5944829817715744, 0.5928331749413545, 0.5917273549270595, 0.5917540486749714, 0.5909378584961376]
         1 hidden LogTrans EXP
          0.5926 0.5892 [0.5936196301698521, 0.5936797722670559, 0.5939045661933922, 0.5927682229403981, 0.589176677540374]


          2hidden No logTrans（None,f,k）->matmul->(None,k,f)->(None,k,1)->matmul->(None,1)
          0.5952 0.5946 [0.5957791425203751, 0.5946420974215796, 0.5948444286504844, 0.5961980021498429, 0.5946272081661552]
          2hidden logTrans
          0.5920 0.5902 [0.5929871897617132, 0.5901566784968396, 0.5932611045846233, 0.5915536980622745, 0.5921170526253287]
          2hidden logTrans EXP
          0.5919 0.5909 [0.5916505895321356, 0.5911030194089643, 0.5913770198162199, 0.5945982006502251, 0.5909147948190725]

'''

#def __init__():

    # if self.use_FM and self.attention_FM:
    # AFM2:DOUBLE ATTENTION
    # self.AFM_weights['attention_W_k']=tf.Variable(initializer([self.c,self.attention_t]))#shape=(c,t)
    # self.AFM_weights['attention_b_k']=tf.Variable(initializer([self.attention_t]))  # shape=(c,t)
    # self.AFM_weights['projection_h_k']=tf.Variable(initializer([self.attention_t,1]))

    # AFM3:DO FINAL K ATTENTION on(None,k)
    # self.AFM_weights['attention_W_finalK']=tf.Variable(initializer([self.k,self.attention_t]))
    # self.AFM_weights['attention_b_finaLK'] = tf.Variable(initializer([self.attention_t]))
    # self.AFM_weights['projection_h_finalK'] = tf.Variable(initializer([self.attention_t,self.k]))



'''
#double attention
def AFM2(self,embedding,AFM_weights):
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
    attention_out = cross_term * self.normalize_att_score  # (None,c,k)*(None,c,1)=(None,c,k)
    #串行Double Attention. 并行的话下行改成cross_term
    k_attention_space=tf.nn.relu(tf.tensordot(attention_out,AFM_weights['attention_W_k'],axes=[[1],[0]])+AFM_weights['attention_b_k'])#注意顺序！(None,$c,k)*($c,t)=(none,k,t) + (t)
    k_att_score=tf.tensordot(k_attention_space,AFM_weights['projection_h_k'],axes=[[2],[0]])#(None,k,$t)*(t,$1)=(None,k,1)
    #k_mask应该是(None,1,k)，而需要的mask是(None.k.1)所以转一下
    k_att_score=tf.transpose(k_att_score,perm=[0,2,1])
    self.k_normalize_att_score=tf.nn.softmax(k_att_score)
    print("self.k_normalize_att_score.shape=",self.k_normalize_att_score.shape)

    attention_out=attention_out*self.k_normalize_att_score
    attention_out=tf.reduce_sum(attention_out,axis=1)#Sum pooling on cross terms. Get (None,k)
    return tf.matmul(attention_out,AFM_weights['projection_p'])#(None,k)*(k,1)=(None,1)

#out None,k, do attention instead of mul p （not make sense）
def AFM3(self, embedding, AFM_weights):
    cross_term = []
    for i in range(self.fields):
        for j in range(i + 1, self.fields):
            # embedding[:,i,:] shape是(None,k)
            if (i, j) in self.FM_ignore_interaction:
                continue
            cross_term.append(embedding[:, i, :] * embedding[:, j, :])  # (None,k) ！！不压缩成(None,1)
    cross_term = tf.stack(cross_term, axis=1)  # (None,c,k)  c=cross term num. tf.stacke add new dim@1
    attention_space = tf.nn.relu(tf.tensordot(cross_term, AFM_weights['attention_W'], axes=[[2], [0]]) + AFM_weights['attention_b'])  # attention_W:(k,t) out:(None,c,t)  +attention_b(t)=(None,c,t)
    att_score = tf.tensordot(attention_space, AFM_weights['projection_h'],axes=[[2], [0]])  # (None,c,t)*(t,1)=(None,c,1)
    self.normalize_att_score = tf.nn.softmax(att_score)  # (None,c,1)
    attention_out = cross_term * self.normalize_att_score  # (None,c,k)*(None,c,1)=(None,c,k)
    attention_out = tf.reduce_sum(attention_out, axis=1)  # Sum pooling on cross terms. Get (None,k)

    k_att_space=tf.nn.relu(tf.matmul(attention_out,AFM_weights['attention_W_finalK'])+AFM_weights['attention_b_finaLK'])#None,t
    k_att_score=tf.matmul(k_att_space,AFM_weights['projection_h_finalK'])#None,t*t,K = None,K
    self.k_normalize_att_score=tf.nn.softmax(k_att_score)#None,k
    k_attention_out=attention_out*self.k_normalize_att_score
    return tf.matmul(k_attention_out, AFM_weights['projection_p'])  # (None,k)*(k,1)=(None,1)
'''
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')