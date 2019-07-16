import numpy as np
import pandas as pd
import matplotlib.pylab as plt
plt.style.use("seaborn")
import seaborn as sns
from matplotlib import gridspec

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Seahorse_buys/acad1Buy3ML10years.txt", delimiter="\s+", header = None)

osc = df.iloc[21:, 0:41].reset_index()
stk = df.iloc[21:, 41:82].reset_index()
macd = df.iloc[21:, 82:123].reset_index()
rtn = df.iloc[21:, 123]
label = np.sign(rtn)
label = label.map({1: 1, -1: 0, 0:0})
results = label.map({1: 'EARN', -1: 'LOSS', 0: 'LOSS'})

# Macd analysis

# Q1. How many zero value of MACD is recorded before a buy signal
# Q2. How long it takes from the most rescent MACD signal to the buy signal
macd_0 = []
macd_timepoint = []
rec_label = []
rec_rtn = []
for i in range(osc.shape[0]):

    data_osc = osc.iloc[i,:]
    data_macd = macd.iloc[i,1:]
    data_label = label.iloc[i]
    data_rtn = rtn.iloc[i]

    # return how many macd values are equal to 0
    macd_0.append(sum(data_macd == 0))

    # return the distance between the signal and nearest time points that macd recorded as 0
    macd_timepoint.append(123 - np.argmax(data_macd == 0))

    rec_label.append(data_label)
    rec_rtn.append(data_rtn)

# construct the final record data frame
df_macd = pd.DataFrame({"macd_0": macd_0,
                       "distance_from_signal": macd_timepoint,
                       "label": rec_label,
                       "rtn":rec_rtn})

# explore the histogram
with plt.style.context("bmh"):
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
    ax1.hist(df_macd["macd_0"])
    ax1.set_title('# of MACD = 0 before signal')
    ax1.set_ylabel('count')


    ax2.hist(df_macd["distance_from_signal"])
    ax2.set_title("distance between signal and nearest MACD = 0")
    ax2.set_ylabel('count')
    plt.tight_layout()

    df_macd_0 = df_macd.loc[df_macd['macd_0'] < 5]
    g1 = sns.FacetGrid(df_macd_0, col="macd_0", col_wrap=6)
    g1.map(plt.hist, "rtn", bins=10)


    df_macd_tp = df_macd.loc[df_macd['distance_from_signal'] > 35]
    g2 = sns.FacetGrid(df_macd_tp, col="distance_from_signal", col_wrap=6)
    g2.map(plt.hist, "rtn", bins=10)


# box plot format from Simon, pretty!

# remove outliers
df_macd_ro = df_macd.loc[df_macd['rtn'] < 25]

# whole plot of all macd = 0 record
with plt.style.context("bmh"):
    fig = plt.figure(figsize = (11.7, 8.27))
    layout = (1,1)
    ax = plt.subplot2grid(layout, (0,0))
    sns.boxplot(ax = ax, x = "macd_0", y = "rtn", data = df_macd_ro)

plt.xlabel("# of MACD = 0 before signal")
plt.ylabel("return value (gain/loss)")


# the top 5 macd = 0 record with most data points
# remove outlier
df_macd_0 = df_macd_0.loc[df_macd_0['rtn'] < 25]

with plt.style.context("bmh"):
    fig = plt.figure(figsize = (11.7, 8.27))
    layout = (1,1)
    ax = plt.subplot2grid(layout, (0,0))
    sns.boxplot(ax = ax, x = "macd_0", y = "rtn", data = df_macd_0)

plt.xlabel("# of MACD = 0 before signal")
plt.ylabel("return value (gain/loss)")

# whole plot of all possible distance
with plt.style.context("bmh"):
    fig = plt.figure(figsize = (11.7, 8.27))
    layout = (1,1)
    ax = plt.subplot2grid(layout, (0,0))
    sns.boxplot(ax = ax, x = "distance_from_signal", y = "rtn", data = df_macd_ro).invert_xaxis()

plt.xlabel("distance between signal and the nearest point for MACD=0")
plt.ylabel("return value (gain/loss)")

# 5 top distance
df_macd_tp = df_macd_tp.loc[df_macd_tp['rtn'] < 25]
with plt.style.context("bmh"):
    fig = plt.figure(figsize = (11.7, 8.27))
    layout = (1,1)
    ax = plt.subplot2grid(layout, (0,0))
    sns.boxplot(ax = ax, x = "distance_from_signal", y = "rtn", data = df_macd_tp).invert_xaxis()

plt.xlabel("distance between signal and the nearest point for MACD=0")
plt.ylabel("return value (gain/loss)")
