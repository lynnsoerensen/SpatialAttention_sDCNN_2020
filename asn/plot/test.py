import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plotPredictions(trial, labels, output_path=None, onset=100, legend=True):
    time_steps = np.arange(-onset, trial.shape[0] - onset)

    sns.set_context("poster")

    sns.set_palette('colorblind')

    plt.figure(figsize=(8.5, 8.5))
    plt.plot(time_steps,trial, linewidth = 7)
    plt.axhline(0.5, color='k', ls = '--',alpha=0.5)
    if legend == True:
        plt.legend(labels, frameon=False, loc = (1.04, 0))
    plt.ylabel('Prediction')
    plt.xlabel('Time (ms)')
    sns.despine()
    if output_path == None:
        plt.show()
    else:
        plt.savefig(output_path)
