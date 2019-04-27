
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