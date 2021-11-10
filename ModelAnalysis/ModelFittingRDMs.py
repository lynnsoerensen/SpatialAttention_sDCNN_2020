import joblib
import os
from scipy.stats import zscore
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, rankdata
from scipy.spatial.distance import squareform
from sklearn.utils import resample
from sklearn.metrics import explained_variance_score

from joblib import Parallel,delayed
import multiprocessing
from pingouin import partial_corr, pcorr, corr
import pandas as pd


def estimateCorr(tracker, cond, num_draws=100, onset=100, path=None):
    from pingouin import pcorr
    idx = np.arange(len(tracker['RDMs']['analog']['corr']))
    results = pd.DataFrame([], columns=['Time', 'Model', 'Draw', 'Correlation - Analog', 'Correlation - Category',
                                        'Partial Correlation - Analog', 'Partial Correlation - Category'])

    if cond != 'neutral':
        splits = cond.split(' - ')
        data = tracker['RDMs'][splits[0]][splits[1]]['corr']
    else:
        data = tracker['RDMs']['neutral']['corr']

    rdms = pd.DataFrame(np.zeros((data.shape[1], 2)),
                        columns=['category', 'analog', ])
    rdms['category'] =tracker['RDMs']['category']
    rdms['analog'] = tracker['RDMs']['analog']['corr']

    counter = 0

    for d in range(num_draws):
        print('Draw ' + str(d))
        if d == 0:
            draw = idx
        else:
            draw = resample(idx, random_state=d)

        for t in range(onset, data.shape[0]):
            #print('Time point ' + str(t))
            # update the dataframe
            rdms[cond] = data[t, :]

            tmp_pcorr = rdms.loc[draw, [cond, 'analog','category']].pcorr()
            tmp_corr = rdms.loc[draw, [cond, 'analog','category']].corr()

            results.loc[counter, 'Time'] = t
            results.loc[counter, 'Model'] = cond
            results.loc[counter, 'Draw'] = d
            results.loc[counter, 'Correlation - Analog'] = tmp_corr.loc['analog', cond]
            results.loc[counter, 'Correlation - Category'] = tmp_corr.loc['category', cond]
            results.loc[counter, 'Partial Correlation - Analog'] = tmp_pcorr.loc['analog', cond]
            results.loc[counter, 'Partial Correlation - Category'] = tmp_pcorr.loc['category', cond]

            counter = counter + 1
        # Save after 100 draws and restart
        if np.mod(d + 1, 100) == 0:
            results.to_pickle(out_path + 'Fits_corr_' + str(d) + '_' + cond + '.pkl')
            # reset dataframe to reduce memory load
            results = pd.DataFrame([],
                                   columns=['Time', 'Model', 'Draw', 'Correlation - Analog', 'Correlation - Category',
                                            'Partial Correlation - Analog', 'Partial Correlation - Category'])
            counter = 0


dir_path = os.path.abspath('')
no_cpu = multiprocessing.cpu_count()
out_path = dir_path + '/ModelAnalysis/RDMFits/'

tracker = joblib.load(out_path + 'RDMs.pkl')

# make categorical RDM
num_img = 50
cat_rdm = np.ones((num_img *len(tracker['categories'] ),num_img *len(tracker['categories'])))
for cat in range(len(tracker['categories'])):
    cat_rdm[cat * num_img: (cat+1) * num_img, cat * num_img: (cat+1) * num_img] = 0


onset = 100

tracker['RDMs']['category'] = squareform(cat_rdm)

conds = ['neutral', 'P-0_I-0.15_O-0 - valid', 'P-0_I-0.15_O-0 - invalid',
         'P-0_I-0_O-0.3 - valid', 'P-0_I-0_O-0.3 - invalid',
         'P-0.45_I-0_O-0 - valid', 'P-0.45_I-0_O-0 - invalid']

Parallel(n_jobs=no_cpu-1)(delayed(estimateCorr)(tracker,cond, num_draws=500,
                                                path=out_path, onset=onset) for cond in conds)


