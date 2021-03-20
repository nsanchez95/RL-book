import numpy as np
import matplotlib.pyplot as plt


trading_pnl_hist_as = np.genfromtxt('trading_pnl_as.csv')
inventory_hist_as = np.genfromtxt('inventory_as.csv')
ob_ask_hist_as = np.genfromtxt('ob_ask_as.csv')
ob_bid_hist_as = np.genfromtxt('ob_bid_as.csv')


trading_pnl_hist_n = np.genfromtxt('trading_pnl_naive.csv')
inventory_hist_n = np.genfromtxt('inventory_naive.csv')
ob_ask_hist_n = np.genfromtxt('ob_ask_naive.csv')
ob_bid_hist_n = np.genfromtxt('ob_bid_naive.csv')

# final_pnl_as = trading_pnl_hist_as[:,-1]
# final_pnl_naive = trading_pnl_hist_n[:,-1]

# bins = np.linspace(0, 120, 100)

# plt.hist(final_pnl_as, bins, alpha=0.5, label='AS Optimal')
# plt.hist(final_pnl_naive, bins, alpha=0.5, label='Naive')
# plt.legend(loc='upper right')
# plt.title("Histogram of Final Returns for both strategies")
# plt.show()

