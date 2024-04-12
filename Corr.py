"""
Scalar Correlation Lab
-------------------------------------------------
The following is based on the article "Anomaly Detection of Time Series Correlations via a Novel Lie Group Structure"
which can be found here: https://onlinelibrary.wiley.com/doi/full/10.1002/sta4.494

Two classes are defined here:
    1. corr2 - This class takes a list or 1-dimensional numpy array of correlation values and computes the average and
    standard deviation of the sampling with respect to the Lie group structure outlined in the cited paper.  There is
    additionally a method stdev_window which takes an optional parameter z as input and computes the upper and lower
    correlation value z*self.stdev away from the mean.

    2. dual_corr_analysis - This class takes time-series samples of two separate random variables and a variables samps
    which refers to the number of consecutive samples to compute correlations over.  Each var_i should be a list of lists
    or a list of 1-dimensional numpy arrays.  Each var_i should have the same number of time-series and each time-series
    should be equal in length.  The user is encouraged to think of these time-series as in the example illustrated in
    the article of stock prices over recurring intervals.  For instance var_i will represent a stock and each time-series
    will represent a consecutive sequence of closing prices over recurring intervals (say 1st quarter over a sequence of
    years).  The methods compute sequences of correlations for time-series of each variable that are paired with each other
    in addition to average, standard deviation, upper and lower bound correlation curves.  There is additionally a method
    corr_anomalies which takes an optional input z and computes anomalies of the original correlation sequences for
    values in the original sequences that are below or above a distance z*self.corr_stdev(t) of the mean at step t.  This
    method does not populate a class variable but is instead left to the user to determine the range in which extreme
    values should be considered anomolous.
"""
import numpy as np
import sys
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

class corr2:
    def __init__(self, corr_seq):
        self.corr_seq = np.array(corr_seq)
        self.seq_length = len(self.corr_seq)
        self.avg = self.corr_avg()
        self.stdev = self.corr_stdev()

    def corr_avg(self):
        return np.tanh(sum(np.arctanh(self.corr_seq))/self.seq_length)

    def corr_stdev(self):
        return np.sqrt(2*sum( (np.arctanh(self.avg) - np.arctanh(self.corr_seq))**2)/(self.seq_length-1))

    def stdev_window(self, z=1):
        lower = np.tanh(np.arctanh(self.avg) - z*self.stdev)
        upper = np.tanh(np.arctanh(self.avg) + z*self.stdev)
        return lower,upper


class dual_corr_analysis:
    def __init__(self, var_1, var_2, samps):
        self.var1 = var_1
        self.var2 = var_2
        self.samps   = samps
        if len(self.var1) != len(self.var2):
            print('Variables have different number of sequences.', file=sys.stderr)

        self.num_series = len(self.var1)
        self.seq_length = len(self.var1[0])-samps
        self.num_cores = cpu_count()

        self.corr_seqs = self.compute_correlations()
        self.corr_obs  = self.create_corr_objects()
        self.corr_avg  = self.corr_avg_seq()
        self.corr_stdev = self.corr_stdev_seq()
        self.corr_lower_seq, self.corr_upper_seq = self.corr_stdev_window()

    def corr_seq(self,x,y):
        if len(x) != len(y):
            print('Sequences are of different lengths.', file = sys.stderr)
        corr = []
        for i in range(len(x) - self.samps + 1):
            corr.append(np.corrcoef(x[i:(i+self.samps)], y[i:(i+self.samps)])[0,1])
        return np.array(corr)

    def compute_correlations(self):
        pairs = [(self.var1[i], self.var2[i]) for i in range(self.num_series)]
        with ThreadPoolExecutor(max_workers=self.num_cores) as executor:
            result = list(executor.map(lambda pair: self.corr_seq(*pair), pairs))
        return result

    def align_samps(self, n):
        time_seq = [self.corr_seqs[i][n] for i in range(self.num_series)]
        return time_seq

    def create_corr_objects(self):
        corr_samps = [self.align_samps(i) for i in range(self.seq_length+1)]
        with ThreadPoolExecutor(max_workers=self.num_cores) as executor:
            result = list(executor.map(corr2, corr_samps))
        return result

    def corr_avg_seq(self):
        avgs = [c.avg for c in self.corr_obs]
        return np.array(avgs)

    def corr_stdev_seq(self):
        stdevs = [c.stdev for c in self.corr_obs]
        return np.array(stdevs)

    def corr_stdev_window(self):
        lower_seq, upper_seq = [], []
        for c in self.corr_obs:
            l,u = c.stdev_window()
            lower_seq.append(l)
            upper_seq.append(u)
        return np.array(lower_seq), np.array(upper_seq)

    def single_time_anomalies(self, n, z=1):
        c_obj = self.corr_obs[n]
        lower, upper = c_obj.stdev_window(z)
        c_upper = []; c_lower = []
        for i in range(self.num_series):
            c = self.corr_seqs[i][n]
            if c >= upper:
                c_upper.append((i,c))
            if c <= lower:
                c_lower.append((i,c))
        anom_dict = {'lower':c_lower, 'upper':c_upper}
        return anom_dict

    def corr_anomalies(self, z=1):
        times = [(i,z) for i in range(len(self.corr_avg))]
        with ThreadPoolExecutor(max_workers=self.num_cores) as executor:
            pre_result = list(executor.map(lambda pair: self.single_time_anomalies(*pair), times))
        result = [(i,pre_result[i]) for i in range(len(pre_result))]
        return result