import tensorflow as tf
import numpy as np
import time

config = tf.ConfigProto(device_count={'GPU': 0})
THREADS=8

ALGO=['LR','FM','DEEP_FM'][2]
MODE=['TRAIN','READ'][0]


TEST_1=False
if TEST_1:
    features, labels = (np.array([np.random.sample((100,2))]),
                        np.array([np.random.sample((100,1))]))
    dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(20)

    iter = dataset.make_one_shot_iterator()
    x, y = iter.get_next()


    # make a simple model
    net = tf.layers.dense(x, 8) # pass the first value from iter.get_next()
    net = tf.layers.dense(net, 8)
    prediction = tf.layers.dense(net, 1)
    loss = tf.losses.mean_squared_error(prediction, y) # pass the second value from iter.get_net() as label
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    N_EPOCHS=50
    sess=tf.Session(config=config)
    #with  as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(N_EPOCHS):
        _, loss_value = sess.run([optimizer, loss])
        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))

    ds=tf.data.TextLineDataset('../data/movie_lens_100k/ua.base')
    ds=ds.repeat().batch(100).prefetch(1024)
    it=ds.make_one_shot_iterator()
    el=it.get_next()
    print(sess.run(el))

    record_defaults=[tf.int32]*3
    ds=tf.data.experimental.CsvDataset('../data/movie_lens_100k/ua.base',record_defaults=record_defaults,header=True,select_cols=[0,1,2],field_delim='\t')
    ds=ds.repeat().batch(100).prefetch(1024)#插入shuffle?
    it=ds.make_one_shot_iterator()
    el=it.get_next()
    print(sess.run(el))
    sess.close()

bs=5000
#libsvm_path=r'C:\Users\13414\Desktop\kkbox_preprocess\data\ctr_demo_join.libsvm'
libsvm_path='../deepx/ft_local/kkbox/data/ctr_demo_join.libsvm'

def decode_libsvm(line):
    lines=tf.reshape(tf.string_split(line, ' ').values,[bs,-1])
    labels = tf.string_to_number(lines[:,0], out_type=tf.float32)
    splits = tf.string_split(tf.reshape(lines[:,1:],[-1]), ':')
    id_vals = tf.reshape(splits.values,splits.dense_shape) #become dense
    feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
    feat_ids = tf.string_to_number(feat_ids, out_type=tf.int64)
    feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)

    feat_ids=tf.reshape(feat_ids,[bs,11])
    feat_vals=tf.reshape(feat_vals,[bs,11])
    return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

def get_input_tensor(libsvm_path,threads):
    ds=tf.data.TextLineDataset(libsvm_path)
    ds=ds.batch(bs).map(decode_libsvm,num_parallel_calls=threads).repeat()
    #ds=ds.prefetch(6000)
    it=ds.make_one_shot_iterator()
    feats,labels=it.get_next()
    #feats_=sess.run(feats)  #100,11,1
    return feats['feat_ids'],tf.expand_dims(labels,axis=-1)# ids:N,f ; labels: N,1


x,y=get_input_tensor(libsvm_path,threads=THREADS)
x=tf.mod(x,1000000)

initializer = tf.contrib.layers.xavier_initializer()

if ALGO=='LR':
    w = tf.Variable(initializer([1000000, 1]))
    b = tf.Variable(initializer([1]))
    z=tf.reduce_sum(tf.reshape(tf.nn.embedding_lookup(w,x), [-1, 11]), axis=1, keepdims=True) + b

elif ALGO=='FM':
    k=8
    w = tf.Variable(initializer([1000000, 1]))
    b = tf.Variable(initializer([1]))
    z=tf.reduce_sum(tf.reshape(tf.nn.embedding_lookup(w,x), [-1, 11]), axis=1, keepdims=True) + b

    embedding_weights = tf.Variable(initializer([1000000, k]))
    embedding=tf.nn.embedding_lookup(embedding_weights,x)
    square_sum = tf.square(tf.reduce_sum(embedding, axis=1))  # (None,k)
    sum_square = tf.reduce_sum(tf.square(embedding), axis=1)  # (None,k)
    cross_term = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=1, keepdims=True)

    z += cross_term

elif ALGO=='DEEP_FM':
    k=8
    w = tf.Variable(initializer([1000000, 1]))
    b = tf.Variable(initializer([1]))
    z=tf.reduce_sum(tf.reshape(tf.nn.embedding_lookup(w,x), [-1, 11]), axis=1, keepdims=True) + b

    embedding_weights = tf.Variable(initializer([1000000, k]))
    embedding=tf.nn.embedding_lookup(embedding_weights,x)

    square_sum = tf.square(tf.reduce_sum(embedding, axis=1))  # (None,k)
    sum_square = tf.reduce_sum(tf.square(embedding), axis=1)  # (None,k)
    cross_term = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=1, keepdims=True)

    fc=tf.keras.layers.Dense(64,activation=tf.nn.relu)(tf.reshape(embedding,[-1,embedding.shape[1]*embedding.shape[2]]))
    fc=tf.keras.layers.Dense(32,activation=tf.nn.relu)(fc)
    fc=tf.keras.layers.Dense(1,activation=tf.nn.relu)(fc)

    z = z + cross_term + fc

else:
    raise Exception("must specific algorithm in %s" % ALGO)

l=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=z))
p=tf.nn.sigmoid(z)
optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(l)

sess=tf.Session(config=config)
sess.run(tf.global_variables_initializer())


if MODE=='TRAIN':
    start_time=time.time()
    lines=1500000
    for i in range(lines//bs):
        try:
            _ = sess.run([optimizer])
        except Exception as e:
            print(str(e)[-400:])

    print('[Algo]',ALGO,'[BatchSize]',bs,'[TRAIN TIME]',round(time.time()-start_time,2),' seconds [Parallel]:',THREADS)

if MODE=='READ':
    start_time=time.time()
    ds = tf.data.TextLineDataset(libsvm_path)
    ds = ds.batch(bs).map(decode_libsvm, num_parallel_calls=THREADS).repeat()  # todo
    lines = 1500000
    it = ds.make_one_shot_iterator()
    el = it.get_next()
    for i in range(lines // bs):
        try:
            _ = sess.run(el)
        except Exception as e:
            print(str(e)[-400:])
    print('[Algo]',ALGO,'[BatchSize]',bs,'[READ TIME]',round(time.time()-start_time,2),' seconds [Parallel]:',THREADS)

#bs 1000 para=8 26s  | pre=1 28.42s | pre=10 27s
#bs 2000 para=8 24.91s~25.4s |pre=2000  25.25s |
#bs 200 55.95s


#bs2000;pre=10;thread=8:  total time= 25s   Readtime=22s|
#bs2000;pre=6000 thread=8 read time=20.5s |  bs2000;pre=10000 t=8 Readtime 20.8 | no prefetch same

#batch -> map : READ:7s Train+Read 10.7s
#LR bs=2000 k8s:
#    t=12 TRAIN:12.63s  READ:8.98
#     t=8 TRAIN:11.97s   READ:8.6s   TRAIN_ONLY:3.4s/epoch
#     t=4 TRAIN:13.77s   READ:8.98s
#     t=1 TRAIN:15.82s  READ:14s
#FM bs=2000 k8s:
#    t=8  TRAIN:31.07s  READ:8.54s   TRAIN_ONLY:22.4s/epoch

#READ SPEED UP LR:
#   bs=5000 TRAIN:10.86s  READ:8.20  TRAIN_ONLY:2.66      =>deepX 2s
#   bs=4000 TRAIN:11.15s  READ:8.19  TRAIN_ONLY:2.96
#   bs=3000 TRAIN:12.53s  READ:8.34  TRAIN_ONLY:4.19
#   bs=2000 TRAIN:12.63s  READ:8.98  TRAIN_ONLY:3.40
#   bs=1000 TRAIN:16.88s  READ:9.26  TRAIN_ONLY:7.62

#READ SPEED UP FM:
#   bs=5000 TRAIN:17.58s  READ:8.20  TRAIN_ONLY:9.37      =>deepX 6s
#   bs=4000 TRAIN:19.26s  READ:8.19  TRAIN_ONLY:11.06
#   bs=3000 TRAIN:23.41s  READ:8.34  TRAIN_ONLY:15.07
#   bs=2000 TRAIN:31.71s  READ:8.98  TRAIN_ONLY:22.73
#   bs=1000 TRAIN:57.15s  READ:9.26  TRAIN_ONLY:47.79

#DEEP_FM
#   bs=5000 TRAIN:20.97   READ:8.20  TRAIN_ONLY:12.77      =>deepX 25s (1thread~4thread same)
#   bs=4000 TRAIN:22.89   READ:8.19  TRAIN_ONLY:14.70


'''

LR:
# 60s  20epoch training
# thread=8 20 epoch same

FM K=8
60s 10epoch

DFM(64,32) K=8
61000 inst/s  24.59s/epoch?   
(16.21.00~16.25.09) 249s 10epoch : 24.9s/epoch   run2(t=4):250s

'''

'''
from tensorflow.python.client import timeline
run_metadata = tf.RunMetadata()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
config = tf.ConfigProto(graph_options=tf.GraphOptions(
        optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)),device_count = {'GPU': 0})
with tf.Session(config=config) as sess:
    c_np = sess.run(optimizer,options=run_options,run_metadata=run_metadata)
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
with open('C:/Users/13414/Desktop/TME/timeline.json','w') as wd:
    wd.write(ctf)
'''