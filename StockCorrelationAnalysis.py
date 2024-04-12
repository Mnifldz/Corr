import numpy as np
import yfinance as yf
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt
import Corr

def source_finance_data(tick1, tick2):
    Stock1 = []; Stock2 = []
    for i in range(num_years):
        print(f'Downloading data from Yahoo Finance {i + 1} of {num_years}')
        Stock1 += [yf.download(tick1,
                            start=str(2015 + i) + '-01-01',
                            end=str(2015 + i) + '-03-31',
                            progress=False)]

        Stock2 += [yf.download(tick2,
                             start=str(2015 + i) + '-01-01',
                             end=str(2015 + i) + '-03-31',
                             progress=False)]

    Var_1 = {'Ticker': tick1, 'prices': [s.Close for s in Stock1]}
    Var_2 = {'Ticker': tick2, 'prices': [s.Close for s in Stock2]}
    return Var_1, Var_2

# Download Finance Data and Compute Correlations
#-------------------------------------------------------------------------
num_years = 9
Tick_1 = 'NVDA'; Tick_2 = 'IBM'
var_1, var_2 = source_finance_data(Tick_1, Tick_2)
C = Corr.dual_corr_analysis(var_1['prices'], var_2['prices'], 5)

# Plot Average and Standard Deviation Curves
#-------------------------------------------------------------------------
x = [i for i in range(C.seq_length+1)]
plt.figure()
plt.plot(x, C.corr_avg, linewidth=0.5)
plt.plot(x, C.corr_lower_seq, color='r', linestyle='-', linewidth=0.5)
plt.plot(x, C.corr_upper_seq, color='r', linestyle='-', linewidth=0.5)
plt.ylim(-1.1, 1.1)
plt.ylabel("Correlation")
plt.xticks([0, 21, 40], ["January", "February", "March"], rotation=70)
plt.title("$\overline{c}_{XY}(t)$ and $\sigma$-Window for " + var_1['Ticker'] + "/" + var_2['Ticker'] + " (First Quarter, 2015 - " + str(2015 + num_years - 1) + ")")
plt.legend(["$\overline{c}_{XY}(t)$", "$\sigma_{XY}$-Window"], loc='upper right', bbox_to_anchor=(1.32, 1))
plt.show()

# Plot Correlation Curves with Anomalies
#-------------------------------------------------------------------------
anomalies = C.corr_anomalies()

fig, axs = plt.subplots(3, 3)
fig.tight_layout()
place = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
for k in range(num_years):
    axs[place[k][0], place[k][1]].plot(x, C.corr_seqs[k][:C.seq_length+1], linewidth=0.5)
    axs[place[k][0], place[k][1]].title.set_text(str(2015 + k))
    plt.sca(axs[place[k][0], place[k][1]])
    feb_0 = min(np.where(var_1['prices'][k].index.month == 2)[0])
    mar_0 = min(np.where(var_1['prices'][k].index.month == 3)[0])
    plt.xticks([0, feb_0, mar_0], ["Jan", "Feb", "Mar"])
    plt.ylim(-1.1, 1.1)

# Plot Anomalies
for t, dct in anomalies:
    lower = dct['lower']; upper = dct['upper']
    if lower:
        l_seq, l_val = lower[0]
        axs[place[l_seq][0], place[l_seq][1]].plot([t], [l_val], color='r', marker='*', linestyle='None', markersize=2)
    if upper:
        u_seq, u_val = upper[0]
        axs[place[u_seq][0], place[u_seq][1]].plot([t], [u_val], color='r', marker='*', linestyle='None', markersize=2)

fig.suptitle(var_1['Ticker'] + "/" + var_2['Ticker'] + " Correlations 1st Quarter (Various Years)")
plt.show()