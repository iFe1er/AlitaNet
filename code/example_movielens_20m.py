import pandas as pd
import numpy as np
import os
from models import LR,FM,MLP,WideAndDeep,DeepFM,FMAndDeep,AFM,NFM,DeepAFM



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

np.random.seed(2019)
data_dir="../data/movie_lens_20m/"
data = pd.read_csv(data_dir+'tags.csv')[['userId','movieId','tag']]

features=['userId','movieId','tag']

 