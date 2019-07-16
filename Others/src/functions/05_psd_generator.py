# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: May 10, 2019
#
# PURPOSE: The script takes in a dataframe and calculate its psd values based on FFT
#
# INPUT:
#     - A dataframe
#`    - NFFT: int, default at 100
#     - name: str, default at 'osc'
#
# OUTPUT:
#     - A psd dataframe

# Packages Required
import pandas as pd
import spectrum
from spectrum import Periodogram
import warnings
warnings.filterwarnings("ignore")

def psd_generator(data, NFFT = 100, name = "osc"):
    freq = []
    for i in range(data.shape[0]):
        data_osc = data.iloc[i,:]
        p = Periodogram(data_osc, NFFT=NFFT)
        temp_list = list(p.psd)
        freq.append(temp_list)
    col_name = []
    for i in range(int(NFFT/2)+1):
        col_name.append("psd"+str(i))

    psd_df = pd.DataFrame(freq, columns=col_name)
    return psd_df
