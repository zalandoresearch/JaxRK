#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 17:33:23 2019

@author: Ingmar Schuster
"""

import numpy as np, scipy as sp, scipy.stats as stats

from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm

from numpy import exp, log, sqrt
from scipy.special import logsumexp
import pandas as pd

from os.path import expanduser, exists, join
import hashlib
import os
import gzip
import pickle


class DataPrep(object):
    @staticmethod
    def get_data(url, md5, preparation_func, pickle_name, *contained_file_paths):
        """
        preparation_func - takes a dictionary with filenames (contained in a ZIP) as keys, file-like objects as values
        """
        if not exists(pickle_name):
            rval = preparation_func(DataPrep.download_check_unpack(url, md5, *contained_file_paths))
            pickle.dump(rval, gzip.GzipFile(pickle_name, 'w'))
        else:
            rval = pickle.load(gzip.GzipFile(pickle_name, 'r'))
        return rval

    @staticmethod
    def download_check_unpack(url, md5, *contained_file_paths):
        from tempfile import TemporaryDirectory, TemporaryFile
        from urllib.request import urlretrieve
        from zipfile import ZipFile, is_zipfile


        file_objs = {}
        with TemporaryDirectory() as tmpdirname:
            if isinstance(url, str):
                (web_file, _) = urlretrieve(url)
                if md5 is not None:
                    hash_md5 = hashlib.md5()
                    with open(web_file, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
                    assert(hash_md5.hexdigest() == md5)

                if is_zipfile(web_file):
                    zip = ZipFile(web_file)
                    for name in contained_file_paths:
                        extr_name = zip.extract(name, tmpdirname)
                        file_objs[name] = open(extr_name, 'r')
                else:
                    file_objs[web_file] = open(web_file, 'r')
            else:
                 for (i, u) in enumerate(url):
                     (web_file, _) = urlretrieve(u)
                     file_objs[contained_file_paths[i]] = web_file
        return file_objs

    @staticmethod
    def prepare_markov_1st(covariates, measurements):
        """
        pepare date for Markovian (1st order) model.

        covariates - the covariates used for prediction
        measurements - the measurements

        returns: (indepvar_t, depvar_t) 
            indepvar_t : covariates at timepoint t concatenated with measurements at timepoint t - 1
            depvar_t : measurements at timepoint t
        """
        assert(len(covariates.shape) == 2 and len(measurements.shape) == 2)
        indepvar = np.concatenate((covariates[1:, :], measurements[:-1, :]), 1)
        depvar = measurements[1:, :]
        return (indepvar, depvar)




class Power(object):

    def __init__(self, root = '~/Documents/datasets/'):
        self.root = expanduser(root)
        if not exists(self.root):
            os.makedirs(self.root)
        pickle_name = join(self.root, 'power_noisy.gzip')
        self.data = DataPrep.get_data("http://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip",
                                      "41f51806846b6b567b8ae701a300a3de",
                                      lambda file_objs: Power.prepare(Power.load_raw(file_objs["household_power_consumption.txt"])),
                                      pickle_name,
                                      "household_power_consumption.txt")
        self.measurement_interval_in_sec = 60

    def get_ihgp_window(self, no):
        beg = ((pd.to_datetime('2006-12-16 17:24:00') - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))
        if no == 0:
            return self.data[np.bitwise_and(self.data.index >= beg + pd.Timedelta('122 days 06:36:00').total_seconds(),
                                     self.data.index <= beg + pd.Timedelta('132 days 06:36:00').total_seconds())]
        elif no == 1:
            return self.data[np.bitwise_and(self.data.index >= beg + pd.Timedelta('617 days 06:36:00').total_seconds(),
                                     self.data.index <= beg + pd.Timedelta('627 days 06:36:00').total_seconds())]
        elif no == 2:
            return self.data[np.bitwise_and(self.data.index >= beg + pd.Timedelta('1340 days 06:36:00').total_seconds(),
                                     self.data.index <= beg + pd.Timedelta('1350 days 06:36:00').total_seconds())]

    @staticmethod
    def load_raw(path_or_filelike, noise = False, epoch = True):
        df = pd.read_csv(path_or_filelike,
                         sep = ';',
                         na_values='?',
                         parse_dates= {'Date_time' : [0, 1]},
                         infer_datetime_format=True)
        if epoch:
            df['epoch'] = df['Date_time'].astype(np.int64) // 10**9
            df.drop('Date_time', 1, inplace=True)
            df.set_index('epoch', inplace=True)
        else:
            df.set_index('Date_time', inplace=True)
        df.dropna(inplace=True)
        return df

    @staticmethod
    def prepare(data, noise = False):
        data.drop(columns=['Global_reactive_power', 'Global_intensity'], inplace=True)
        if noise:
            # Add noise
            N = len(data)
            rng = np.random.RandomState(42)
            data.Voltage = data.Voltage + 0.01*rng.rand(N, 1)
            data.Global_active_power = data.Global_active_power + 0.001*rng.rand(N, 1)
            for idx in range(1,4):
                name = 'Sub_metering_%d'%idx
                data[name] = data[name] + rng.rand(N, 1)
                data[name] = (data[name] - data[name].mean())/data[name].std()
            for name in ['Voltage', 'Global_active_power']:
                data[name] = (data[name] - data[name].mean())/data[name].std()

        return data

class Traffic(object):

    def __init__(self, root = '~/Documents/datasets/'):
        self.root = expanduser(root)
        if not exists(self.root):
            os.makedirs(self.root)
        pickle_name = join(self.root, 'traffic.gzip')
        self.data = DataPrep.get_data("https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip",
                                      None,
                                      Traffic.load_raw,
                                      pickle_name,
                                      "PEMS_train", "PEMS_trainlabels", "PEMS_test", "PEMS_testlabels")
        self.measurement_interval_in_sec = 600

    @staticmethod
    def load_raw(file_objs):
        colnames = ['DoW', 'Sec of Day']
        colnames.extend(range(963))
        rval = {}
        for split in ('train', 'test'):
            # load labels from raw
            l = np.array(list(map(int, file_objs['PEMS_' + split + 'labels'].readline()[1:-2].split())))

            # load data from raw
            d =  np.array([[list(map(float, r.split()))
                                for r in l[1:-2].split(';')]
                                    for l in file_objs['PEMS_' + split].readlines()])
            d =  np.swapaxes(d, 1, 2) # we want timepoints in rows

            rval[split] = pd.DataFrame(np.hstack((np.repeat(l, d.shape[1])[:, None] - 1,
                                                  np.tile(np.arange(d.shape[1]) * 600, d.shape[0])[:, None],
                                                  np.vstack(d))),
                                       columns = colnames)
        return rval
    
    @staticmethod
    def diff(df):
        left = df[['DoW', 'Sec of Day']][np.mod(np.arange(df.index.size), 144) != 0]
        right = (df.values[1:, 2:] - df.values[:-1, 2:])[np.mod(np.arange(1,df.index.size), 144) != 0, :]
        return pd.DataFrame(np.hstack([left.values, right]), columns = df.columns)

    @staticmethod
    def equalize_smp_size(df, num = None):
        if num is None:
            num = df['DoW'].value_counts().min()
        return df.groupby('DoW').head(num).reset_index(drop=True)

    @staticmethod
    def stackup_smp(df):
        inp = []
        outp = []
        for g in df.groupby(['DoW', 'Sec of Day']):
            inp.append(g[0])
            outp.append(g[1].values[:, 2:])
        print(len(inp))
        return np.array(inp), np.swapaxes(np.stack(outp), 1, 2)
    
    def plot(self, split = "train", dow = [0,1], sensor = 3, bins = [144,60], filename = None):
        import matplotlib.pyplot as plt
        import calendar
        
        xticks = np.arange(0, 144, 20)
        if type(dow) == int:
            values = [Traffic.stackup_smp(Traffic.equalize_smp_size(self.data[split]))[1][dow * 144:(dow+1)*144, sensor:(sensor + 1), :]]
            dow = [dow]
            if False:
                plt.hist2d(np.repeat(np.arange(out.shape[0]), out.shape[2], 0), out.flatten(), bins = bins)
                plt.xlabel("time of day on %s" % calendar.day_name[(dow+6)%7])
                plt.ylabel("road location occupancy")
                plt.tight_layout()
                if filename is not None:
                    plt.savefig(filename, dpi=300)
                plt.show()
        else:
            values = [Traffic.stackup_smp(Traffic.equalize_smp_size(self.data[split]))[1][day * 144:(day+1)*144, sensor:(sensor + 1), :] for day in dow]
        if True:
            fig, ax = plt.subplots(1,len(values), figsize=(6*len(values),3), sharey = True)
            if len(values) == 1:
                ax = [ax]
            for (i, out) in enumerate(values):
                ax[i].hist2d(np.repeat(np.arange(out.shape[0]), out.shape[2], 0), out.flatten(), bins = bins, range = [[0, 143], [0, 0.25]])
                ax[i].set_xlabel("time of day on %s" % calendar.day_name[(dow[i]+6)%7])
                if i is 0:
                    ax[i].set_ylabel("road location occupancy")
            fig.tight_layout()
            if filename is not None:
                fig.savefig(filename, dpi=300)
            fig.show()


class Mountain(object):
    def __init__(self, root = '~/Documents/datasets/'):
        self.root = expanduser(root)
        if not exists(self.root):
            os.makedirs(self.root)
        pickle_name = join(self.root, 'mountain.gzip')
        self.data = DataPrep.get_data("https://github.com/ericlee0803/GP_Derivatives/raw/master/demos/Maps/MtSH.mat",
                                      None,
                                      Mountain.load_raw,
                                      pickle_name)
        self.measurement_interval_in_sec = 600

    @staticmethod
    def load_raw(file_objs):
        mt = sp.io.loadmat(list(file_objs.keys())[0])
        nx, ny = mt['nx'][0,0], mt['ny'][0,0]
        return {"y": mt['mth_points'].T[0].reshape(ny, nx),
                "x": mt['mth_points'].T[1].reshape(ny, nx),
                "z": mt['mth_verts'].reshape(ny, nx),
                "fin_diff": mt['mth_grads'].reshape(ny, nx, 2)}

    def plot_contour(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize= (5, 6))
        c = plt.contour(self.data['x'], self.data['y'], self.data['z'], 40, cmap='winter')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('MtStHelen_contour.pdf')
        plt.show()


class Jura(object):

    def __init__(self, root = '~/Documents/datasets/'):
        self.root = expanduser(root)
        if not exists(self.root):
            os.makedirs(self.root)
        pickle_name = join(self.root, 'jura.gzip')
        self.data = DataPrep.get_data("https://drive.google.com/uc?export=download&id=189FhW0h8iz7BQSDyXH0QSL3cFFFyy9Ln",
                                      None,
                                      Jura.load_csv,
                                      pickle_name,
                                      "All_data.csv")

    @staticmethod
    def load_csv(file_objs):
        return pd.read_csv(file_objs['All_data.csv'], index_col = 0)
    

class ToyExample(object):
    def nonlinear_forward_mean(self, x):
        if len(x.shape) == 2:
            return np.stack(np.apply_along_axis(self.nonlinear_forward_mean, 1, x), axis = 0).reshape(x.shape[0],x.shape[1]*2)
        #return np.vstack((np.sin(x*4)+np.sin(x*3)*2, np.cos(x*2)+ np.cos(x*3)+3)).T
        return np.vstack((np.sin(x), np.cos(x))).T
    
    def nonlinear_forward_sd(self, x):
        if len(x.shape) == 2:
            return np.stack(np.apply_along_axis(self.nonlinear_forward_sd, 1, x), axis = 0).reshape(x.shape[0],x.shape[1]*2)
        #return np.vstack((x**2, exp(np.sin(x)))).T+0.5
        return np.vstack((np.sin(x**2), np.cos(x**2))).T+0.5
    
    def draw_conditioned_on(self, x, samps_per_input):
        means = self.nonlinear_forward_mean(x)[:,:, None]
        sds = self.nonlinear_forward_sd(x)[:,:, None]
        return means + np.random.randn(means.size * samps_per_input).reshape(means.shape[0], means.shape[1], samps_per_input) * sds
    
    def cond_distr(self, inp):        
        assert(len(inp.shape) == 1)
        return multivariate_normal(self.nonlinear_forward_mean(inp).flatten(), np.diag(self.nonlinear_forward_sd(inp).flatten())) 
    
    def cond_prob(self, output, inp):
        return self.cond_distr(inp).pdf(out)
    
    def __init__(self, nsamps_x = 500, nsamps_cond = 20):
        self.x = np.random.randn(nsamps_x, 1) * 3
        self.y = self.nonlinear_forward_mean(self.x)
        self.y = self.y[:,:, None] + np.random.randn(self.y.size*nsamps_cond).reshape(self.y.shape[0], self.y.shape[1], -1) * self.nonlinear_forward_sd(self.x)[:,:, None]
        