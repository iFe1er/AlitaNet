from Alita_DeepFM import Alita_DeepFM
import tensorflow as tf

class LR():
    def __init__(self,features_sizes,loss_type='rmse',hash_size=None,metric_type=None):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,use_LR=True,use_FM=False,use_MLP=False,hash_size=hash_size,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        return self.model.predict(ids_pred)

class MLR():
    def __init__(self,features_sizes,loss_type='rmse',MLR_m=4,hash_size=None,metric_type=None):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,use_LR=False,use_MLR=True,MLR_m=MLR_m,use_FM=False,use_MLP=False,hash_size=hash_size,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        return self.model.predict(ids_pred)

class FM():
    def __init__(self,features_sizes,loss_type='rmse',k=10,FM_ignore_interaction=None,dropout_keeprate=1.0,hash_size=None,metric_type=None):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,k=k,use_LR=True,use_FM=True,use_MLP=False,FM_ignore_interaction=FM_ignore_interaction,dropout_keeprate=dropout_keeprate,hash_size=hash_size,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        return self.model.predict(ids_pred)

class AFM():
    def __init__(self,features_sizes,loss_type='rmse',k=10,FM_ignore_interaction=None,attention_FM=10,dropout_keeprate=1.0,lambda_l2=0.0,metric_type=None):#todo lambda_L2 only added for AFM
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,k=k,use_LR=True,use_FM=True,use_MLP=False,FM_ignore_interaction=FM_ignore_interaction,attention_FM=attention_FM,dropout_keeprate=dropout_keeprate,lambda_l2=lambda_l2,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        return self.model.predict(ids_pred)

class NFM():
    def __init__(self,features_sizes,loss_type='rmse',k=10,dropout_keeprate=1.0,metric_type=None):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,k=k,use_LR=True,use_FM=True,use_MLP=False,use_NFM=True,dropout_keeprate=dropout_keeprate,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        return self.model.predict(ids_pred)

class BiFM():
    def __init__(self,features_sizes,loss_type='rmse',k=10,dropout_keeprate=1.0,metric_type=None):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,k=k,use_LR=True,use_FM=True,use_MLP=False,use_NFM=False,use_BiFM=True,dropout_keeprate=dropout_keeprate,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        return self.model.predict(ids_pred)

class DeepBiFM():
    def __init__(self,features_sizes,dense_features_size=0,loss_type='rmse',deep_layers=(256,256),activation=tf.nn.relu,k=10,dropout_keeprate=1.0,metric_type=None):
        self.model=Alita_DeepFM(features_sizes=features_sizes,dense_features_size=dense_features_size,loss_type=loss_type,k=k,deep_layers=deep_layers,activation=activation,use_LR=True,use_FM=True,use_MLP=True,use_NFM=False,use_BiFM=True,dropout_keeprate=dropout_keeprate,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        return self.model.predict(ids_pred)

class FiBiFM():
    def __init__(self,features_sizes,loss_type='rmse',k=10,dropout_keeprate=1.0,metric_type=None):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,k=k,use_LR=True,use_FM=True,use_MLP=False,use_NFM=False,use_BiFM=True,use_SE=True,dropout_keeprate=dropout_keeprate,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,dense_train=None,dense_test=None,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,dense_train=dense_train,dense_test=dense_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred,dense_pred=None):
        return self.model.predict(ids_pred,dense_pred=dense_pred)

class FiBiNet():
    def __init__(self,features_sizes,loss_type='rmse',k=10,dropout_keeprate=1.0,metric_type=None):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,k=k,use_LR=True,use_FM=True,use_MLP=False,use_NFM=False,use_BiFM=True,use_SE=True,use_FiBiNet=True,dropout_keeprate=dropout_keeprate,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        return self.model.predict(ids_pred)

class MLP():
    def __init__(self,features_sizes,dense_features_size=0,loss_type='rmse',deep_layers=(256,256),activation=tf.nn.relu,k=10,dropout_keeprate=1.0,metric_type=None):
        self.model=Alita_DeepFM(features_sizes=features_sizes,dense_features_size=dense_features_size,loss_type=loss_type,k=k,deep_layers=deep_layers,activation=activation,use_LR=False,use_FM=False,use_MLP=True,dropout_keeprate=dropout_keeprate,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,dense_train=None,dense_test=None,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,dense_train=dense_train,dense_test=dense_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred,dense_pred=None):
        return self.model.predict(ids_pred,dense_pred=dense_pred)


class WideAndDeep():
    def __init__(self,features_sizes,loss_type='rmse',deep_layers=(256,256),activation=tf.nn.relu,k=10,dropout_keeprate=1.0,metric_type=None):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,k=k,deep_layers=deep_layers,activation=activation,use_LR=True,use_FM=False,use_MLP=True,dropout_keeprate=dropout_keeprate,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        return self.model.predict(ids_pred)

class FMAndDeep():
    def __init__(self,features_sizes,loss_type='rmse',deep_layers=(256,256),activation=tf.nn.relu,k=10,FM_ignore_interaction=None,dropout_keeprate=1.0,metric_type=None):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,k=k,deep_layers=deep_layers,activation=activation,use_LR=False,use_FM=True,use_MLP=True,FM_ignore_interaction=FM_ignore_interaction,dropout_keeprate=dropout_keeprate,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        return self.model.predict(ids_pred)

class DeepFM():
    def __init__(self,features_sizes,dense_features_size=0,loss_type='rmse',deep_layers=(256,256),activation=tf.nn.relu,k=10,FM_ignore_interaction=None,dropout_keeprate=1.0,metric_type=None):
        self.model=Alita_DeepFM(features_sizes=features_sizes,dense_features_size=dense_features_size,loss_type=loss_type,k=k,deep_layers=deep_layers,activation=activation,use_LR=True,use_FM=True,use_MLP=True,FM_ignore_interaction=FM_ignore_interaction,dropout_keeprate=dropout_keeprate,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,dense_train=None,dense_test=None,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,dense_train=dense_train,dense_test=dense_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred,dense_pred=None,):
        return self.model.predict(ids_pred,dense_pred=dense_pred)


class DeepAFM():
    def __init__(self,features_sizes,dense_features_size=0,loss_type='rmse',deep_layers=(256,256),activation=tf.nn.relu,k=10,FM_ignore_interaction=None,attention_FM=10,dropout_keeprate=1.0,metric_type=None):
        self.model=Alita_DeepFM(features_sizes=features_sizes,dense_features_size=dense_features_size,loss_type=loss_type,k=k,deep_layers=deep_layers,activation=activation,use_LR=True,use_FM=True,use_MLP=True,FM_ignore_interaction=FM_ignore_interaction,attention_FM=attention_FM,dropout_keeprate=dropout_keeprate,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,dense_train=None,dense_test=None,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,dense_train=dense_train,dense_test=dense_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred,dense_pred=None):
        return self.model.predict(ids_pred,dense_pred=dense_pred)


class AutoInt():
    def __init__(self,features_sizes,loss_type='rmse',k=10,FM_ignore_interaction=None,dropout_keeprate=1.0,autoint_params=None,metric_type=None):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,k=k,use_LR=False,use_FM=False,use_MLP=False,use_AutoInt=True,FM_ignore_interaction=FM_ignore_interaction,autoint_params=autoint_params,dropout_keeprate=dropout_keeprate,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        return self.model.predict(ids_pred)

class DeepAutoInt():
    def __init__(self,features_sizes,dense_features_size=0,loss_type='rmse',k=10,deep_layers=(256,256),activation=tf.nn.relu,FM_ignore_interaction=None,dropout_keeprate=1.0,autoint_params=None,metric_type=None):
        self.model=Alita_DeepFM(features_sizes=features_sizes,dense_features_size=dense_features_size,loss_type=loss_type,k=k,deep_layers=deep_layers,activation=activation,use_LR=True,use_FM=False,use_MLP=True,use_AutoInt=True,FM_ignore_interaction=FM_ignore_interaction,autoint_params=autoint_params,dropout_keeprate=dropout_keeprate,metric_type=metric_type)

    def fit(self,ids_train,ids_test,y_train,y_test,dense_train=None,dense_test=None,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,dense_train=dense_train,dense_test=dense_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred,dense_pred=None,):
        return self.model.predict(ids_pred,dense_pred=dense_pred)