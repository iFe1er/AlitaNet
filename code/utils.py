import numpy as np

def batcher(X_, y_=None, batch_size=-1,hash_size=None):
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
    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:upper_bound]
            yield (ret_x, ret_y)
        else:
            yield ret_x

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

def isBetter(a,b,is_better_greater=True):
    if is_better_greater:
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

