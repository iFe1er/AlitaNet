from Alita_DeepFM import Alita_DeepFM
import tensorflow as tf

class LR():
    def __init__(self,features_sizes,loss_type='rmse'):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,use_LR=True,use_FM=False,use_MLP=False)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        self.model.predict(ids_pred)


class FM():
    def __init__(self,features_sizes,loss_type='rmse',k=10):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,k=k,use_LR=True,use_FM=True,use_MLP=False)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        self.model.predict(ids_pred)


class MLP():
    def __init__(self,features_sizes,loss_type='rmse',deep_layers=(256,256),activation=tf.nn.relu,k=10):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,k=k,deep_layers=deep_layers,activation=activation,use_LR=False,use_FM=False,use_MLP=True)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        self.model.predict(ids_pred)


class WideAndDeep():
    def __init__(self,features_sizes,loss_type='rmse',deep_layers=(256,256),activation=tf.nn.relu,k=10):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,k=k,deep_layers=deep_layers,activation=activation,use_LR=True,use_FM=False,use_MLP=True)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        self.model.predict(ids_pred)

class FMAndDeep():
    def __init__(self,features_sizes,loss_type='rmse',deep_layers=(256,256),activation=tf.nn.relu,k=10):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,k=k,deep_layers=deep_layers,activation=activation,use_LR=False,use_FM=True,use_MLP=True)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        self.model.predict(ids_pred)

class DeepFM():
    def __init__(self,features_sizes,loss_type='rmse',deep_layers=(256,256),activation=tf.nn.relu,k=10):
        self.model=Alita_DeepFM(features_sizes=features_sizes,loss_type=loss_type,k=k,deep_layers=deep_layers,activation=activation,use_LR=True,use_FM=True,use_MLP=True)

    def fit(self,ids_train,ids_test,y_train,y_test,lr=0.001,N_EPOCH=50,batch_size=200,early_stopping_rounds=20):
        return self.model.fit(ids_train,ids_test,y_train,y_test,lr=lr,N_EPOCH=N_EPOCH,batch_size=batch_size,early_stopping_rounds=early_stopping_rounds)

    def predict(self, ids_pred):
        self.model.predict(ids_pred)