import pandas as pd
import numpy as np
import random

import queue
from threading import Thread, Event

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def enqueue(queue, stop, gen_func):
    gen = gen_func()
    while True:
        if stop.is_set():
            return

        queue.put(next(gen))

class data_generators():
    def __init__(self, bz, dataframes, num_class, preprocess):
        '''
        bz:
            batch size
        dataframes:
            dataframes for generators
            should inpt in the format
            
            [
             [kind1_train, kind1_val, kind1_test],
             [kind2_train, kind2_val, kind2_test]
            ]
        num_class:
            number classes to classfier
        preprocess:
            preprocess function
            will receive a dataframe
            and should out put in the format
            [x1, x2], y, file/id 
        
        '''
        
        self.bz = bz
        
        dataframes = list(zip(*dataframes))
        self.train_dfs = dataframes[0]
        self.val_dfs = dataframes[1]
        self.test_dfs = dataframes[2]
        
        self.num_class = num_class
        
        self.preprocess = preprocess
        
    def get_train_idx(self):
        len_list = [len(df) for df in self.train_dfs]
        
        bz_t = self.bz//len(len_list)
        batch_num = [x//bz_t for x in len_list]

        batch_nth = [0] * len(len_list)

        select = [list(range(x)) for x in len_list]

        for s in select:
            random.shuffle(s)

        while True:
            idxs = []
            for i in range(len(len_list)):
                if batch_nth[i] >= batch_num[i]:
                    batch_nth[i] = 0
                    random.shuffle(select[i])
                idx = select[i][batch_nth[i]*bz_t:(batch_nth[i]+1)*bz_t]
                batch_nth[i] += 1
                idxs.append(idx)

            yield idxs
    
    def get_train_data(self):
        while True:
            idxs = self.train_idx_queue.get()

            select_list = []
            for df, idx in zip(self.train_dfs, idxs):
                select_list.append(df.iloc[idx])

            train_list = pd.concat(select_list).sample(frac=1)

            x, y, _ = self.preprocess(train_list, self.num_class, aug=True)

            yield x,  y
    
    def start_train_threads(self):
        '''
        jobs:number of threads
        '''
        
        self.train_queue = queue.Queue(maxsize = 3)
        self.train_idx_queue =queue.Queue(maxsize = 10)
        self.jobs = 1

        ### for stop threads after training ###
        self.events=[]

        ### enqueue train index ###
        event = Event()
        thread = Thread(target = enqueue, args = (self.train_idx_queue, event, self.get_train_idx))
        thread.start()
        self.events.append(event)

        ### enqueue train batch ###
        for i in range(self.jobs):
            event = Event()
            thread = Thread(target = enqueue,args = (self.train_queue, event, self.get_train_data))
            thread.start()
            self.events.append(event)

        
    #### val ####
    def load_val(self):
        val_list = pd.concat(self.val_dfs).sample(frac=1)
        x, y, _ = self.preprocess(val_list, self.num_class)

        self.val_x = list(zip(*[chunks(i, self.bz) for i in x]))
        self.val_y = list(chunks(y, self.bz))

        self.val_len = len(self.val_y)
    
    def iter_val(self):
        for i in range(self.val_len):
            yield self.val_x[i], self.val_y[i], len(self.val_y[i])
    
    #### test ####
    def get_test_data(self):
        test_list = pd.concat(self.test_dfs)
        test_list = chunks(test_list, self.bz)

        for data in test_list:
            x, y, files = self.preprocess(data, self.num_class)

            yield x, y, files