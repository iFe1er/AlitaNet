import numpy as np
import pandas as pd
import tensorflow as tf

def batcher(X_, y_=None,X_dense=None, batch_size=-1,hash_size=None):
    n_samples = X_.shape[0]
    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))
    if hash_size is not None:
        X_hash=list(X_.values)
        for b in X_hash:
            for i in range(len(b)):
                b[i]=abs(hash('key_'+str(i)+'_value_'+str(b[i])))%hash_size
        X_=np.array(X_hash)
        #print(X_)
        #print(y_) if y_ is not None else print(None)
    #只有离散输入
    if X_dense is None:
        for i in range(0, n_samples, batch_size):
            upper_bound = min(i + batch_size, n_samples)
            ret_x = X_[i:upper_bound]
            ret_y = None
            if y_ is not None:
                ret_y = y_[i:upper_bound]
                yield (ret_x,None,ret_y)
            else:
                yield (ret_x,None)
    else:
        for i in range(0, n_samples, batch_size):
            upper_bound = min(i + batch_size, n_samples)
            ret_x = X_[i:upper_bound]
            ret_xdense=X_dense[i:upper_bound]
            ret_y = None
            if y_ is not None:
                ret_y = y_[i:upper_bound]
                yield (ret_x,ret_xdense,ret_y)
            else:
                yield (ret_x,ret_xdense)

#Xs_:[Xs,Xs...] 长度为field
def list_batcher(Xs_, y_=None, batch_size=-1):
    n_samples = Xs_[0].shape[0]
    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = [Xs_[field_id][i:upper_bound] for field_id in range(len(Xs_))]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)
        else:
            yield ret_x

def _build_regression_signature(input_tensor, output_tensor):
  """Helper function for building a regression SignatureDef."""
  input_tensor_info = tf.saved_model.utils.build_tensor_info(input_tensor)
  signature_inputs = {
      tf.saved_model.signature_constants.REGRESS_INPUTS: input_tensor_info
  }
  output_tensor_info = tf.saved_model.utils.build_tensor_info(output_tensor)
  signature_outputs = {
      tf.saved_model.signature_constants.REGRESS_OUTPUTS: output_tensor_info
  }
  return tf.saved_model.signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      tf.saved_model.signature_constants.REGRESS_METHOD_NAME)

def _build_classification_signature(input_tensor, scores_tensor):
  """Helper function for building a classification SignatureDef."""
  input_tensor_info = tf.saved_model.utils.build_tensor_info(input_tensor)
  signature_inputs = {
      tf.saved_model.signature_constants.CLASSIFY_INPUTS: input_tensor_info
  }
  output_tensor_info = tf.saved_model.utils.build_tensor_info(scores_tensor)
  signature_outputs = {
      tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
          output_tensor_info
  }
  return tf.saved_model.signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)


def isBetter(a,b,is_greater_better=True):
    if is_greater_better:
        return a>b
    else:
        return a<b

def FGE_scheduler(epoch,N_EPOCH=100,M=5):
    save_every_step=N_EPOCH/M
    t=(epoch%save_every_step)/save_every_step
    alpha1=0.05;alpha2=0.01
    alpha1=alpha1*(0.8)**int(epoch/20)#new add
    print(alpha1)
    if t<=1/2:#1~10
        return (1-2*t)*alpha1+2*t*alpha2
    else:
        return (2-2*t)*alpha2+(2*t-1)*alpha2

class ColdStartEncoder():
    def __init__(self):
        self.status='init'
        self.encoding_dict=dict()

    def fit(self,col):
        if isinstance(col,pd.Series):
            unique_values=col.unique()
        elif isinstance(col,list):
            unique_values=pd.Series(col).unique()
        else:
            raise Exception('Only Series and list supported')
        #编码从1起步
        self.encoding_dict={value:encoded for value,encoded in zip(unique_values,range(1,len(unique_values)+1))}
        self.status = 'fitted'

    def transform(self,col):
        if self.status!='fitted':
            raise Exception('must fit before transform')
        return col.map(lambda x:self.encoding_dict.get(x)).fillna(0).astype(int).tolist()

    def fit_transform(self,col):
        self.fit(col)
        return self.transform(col)

def multihot_padder(col,sep='|',padding_len=None):
    assert isinstance(col,pd.Series)

    if not padding_len:
        t = col.apply(lambda x: np.array([int(i) for i in x.split('|')]))
        lens = np.array([len(i) for i in t])
        padding_len=max(lens)
    else:
        t = col.apply(lambda x: np.array([int(i) for i in x.split('|')][:padding_len]))
        lens = np.array([len(i) for i in t])

    print("Padding Len: %s"%padding_len)
    mask=np.arange(padding_len)<lens.reshape([-1,1])
    result=np.zeros([col.shape[0],padding_len])
    result[mask]=np.concatenate(t.values)#变成一位向量 填入 print(result[14,:])
    return result,padding_len

'''
#test speed 
from datetime import datetime
startTime = datetime.now()
for i in range(100):#10
    #_=X_train['song_id'].value_counts().index.tolist() #17.6s
    #_=X_train['song_id'].unique().tolist()             #6.64s
endTime = datetime.now()
print('Cost:',endTime - startTime)

'''