from numpy import *
import numpy.ma as ma
import scipy.io as sio
from scipy import stats
from scipy import signal
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas.plotting._converter as pandacnv   # only necessary due to Pandas 0.21.0 bug with Datetime plotting
pandacnv.register()
set_printoptions(threshold=100)   # faster

# filepath
data_dir = '/Users/Ethan/Documents/UW/By course/2018-01 - ATM S 552 (Dennis L. Hartmann)/' \
           '2018-03-05 - assignment 5 on TAO SST/'

# load hourly SST data from TAO array mooring at 0°N, 165°E
# obtained from https://www.pmel.noaa.gov/tao/drupal/disdel/, with gaps filled by DH using climatology
# period supposed to be March 30, 1991 to February 10, 1996, but actually July 13, 1996 to March 18, 1999
data_dict = sio.loadmat(data_dir + 'SST_0N165E.mat')
sst = data_dict['tr'].T[0]
year = data_dict['yr'].T[0]
month = data_dict['mn'].T[0]
day = data_dict['dy'].T[0]
hour = data_dict['hr'].T[0]
dates = [datetime(year[t],month[t],day[t],hour[t]) for t in range(len(sst))]
data = pd.Series(index=dates,data=sst)

# convert from GMT to local time (GMT+11)
data.index = data.index + timedelta(hours=11)

# detrend data
[slope,intercept,_,p] = stats.linregress(mdates.date2num(data.dropna().index.date) + data.dropna().index.hour/24,
                                         data.dropna().values)[0:4]
trend = intercept + (slope * (mdates.date2num(data.index.date) + data.dropna().index.hour/24))
data_detrended = (data - pd.Series(data=trend,index=data.index)) + data.mean()

# plot original and detrended data
fig = plt.figure(figsize=(7.5,3))
plt.plot(data,c='k',lw=0.5,label='Original data (slope = {0:.02f}°C/year, p = {1:.02f})'.format(slope*365,p))
plt.plot(data_detrended,c='b',lw=0.5,label='Detrended data')
plt.ylim([plt.ylim()[0],plt.ylim()[1]+1])
plt.legend(ncol=2,loc='upper right',frameon=False)
plt.title('Data from TAO array mooring at 0°N, 165°E')
plt.ylabel('SST (°C)')
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(data_dir + 'fig_1_trend.pdf')
plt.close()

# switch
data = data_detrended

# create diurnal composite and calculate standard error
diurnal = data.groupby(data.index.hour).mean()
diurnal_std_error = data.groupby(data.index.hour).std() / sqrt(data.groupby(data.index.hour).count())
diurnal_range = diurnal.max() - diurnal.min()

# plot diurnal composite
fig = plt.figure(figsize=(7.5,3.0))
plt.plot(diurnal,c='k',lw=1,zorder=2,label='Composite mean')
plt.fill_between(diurnal.index,diurnal-1.96*diurnal_std_error,diurnal+1.96*diurnal_std_error,
                 facecolor='k',alpha=0.3,zorder=1,label='±1.96xSE (95% confidence)')
plt.legend(loc='upper left',frameon=False)
plt.xlim([0,23])
plt.xlabel('Hour of day in local time (GMT+11)')
plt.title('Diurnal composite (range = {0:.2f}°C, min = hour {1}, max = hour {2})'
          .format(diurnal_range,diurnal.index[diurnal.idxmin()],diurnal.index[diurnal.idxmax()]))
plt.tight_layout()
plt.savefig(data_dir + 'fig_2_diurnal.pdf')
plt.close()

# Fourier expansion of diurnal cycle
y = array(diurnal.values)
t = array(diurnal.index)
N = len(diurnal)
i = arange(N)
delta_T = 1
T = N*delta_T
harmonics = zeros((4,N))
coeff = zeros((4,2))
amplitudes = zeros(4)
for k in range(4):
    coeff[k,:] = [(2/N) * sum(y * cos(2*pi*(k+1)*i/N)), (2/N) * sum(y * sin(2*pi*(k+1)*i/N))]
    harmonics[k,:] = coeff[k,0] * cos(2*pi*(k+1)*t/T) + coeff[k,1] * sin(2*pi*(k+1)*t/T)
    amplitudes[k] = sqrt(coeff[k,0]**2 + coeff[k,1]**2)
    # diurnal_residual = diurnal_residual - harmonics[k,:]    # thought might be necessary, but wasn't

# plot Fourier expansion of diurnal cycle
fig = plt.figure(figsize=(7.5,4.0))
plt.plot(diurnal,c='k',alpha=0.6,label='Diurnal composite')
plt.plot(t,diurnal.mean() + harmonics[0,:],c='red',
         label='k = 1 (diurnal); C = {0:.03f}°C'.format(amplitudes[0]))
plt.plot(t,diurnal.mean() + harmonics[1,:],c='blue',
         label='k = 2 (semi-diurnal); C = {0:.03f}°C, {1:.0f}% of k = 1'.format(amplitudes[1],
                                                                                100*amplitudes[1]/amplitudes[0]))
plt.plot(t,diurnal.mean() + harmonics[2,:],c='orange',
         label='k = 3 (ter-diurnal); C = {0:.03f}°C, {1:.0f}% of k = 1'.format(amplitudes[2],
                                                                               100*amplitudes[2]/amplitudes[0]))
plt.plot(t,diurnal.mean() + harmonics[3,:],c='green',
         label='k = 4 (quadra-diurnal); C = {0:.03f}°C, {1:.0f}% of k = 1'.format(amplitudes[3],
                                                                                  100*amplitudes[3]/amplitudes[0]))
plt.plot(t,diurnal.mean() + harmonics.sum(axis=0),c='k',lw=2.5,ls='--',alpha=0.6,label='Sum of k = 1 to 4')
plt.ylim([plt.ylim()[0],plt.ylim()[1]+0.025])
plt.legend(ncol=1,loc='upper left',fontsize=8,frameon=False)
plt.xlim([0,23])
plt.xlabel('Hour of day in local time (GMT+11)')
plt.ylabel('SST (°C)')
plt.tight_layout()
plt.savefig(data_dir + 'fig_3_diurnal_fourier.pdf')
plt.close()

# insolation function (flipped sign from DH formula)
hours = arange(24)
ins = -1*cos(2*pi*hours/24)
ins[0:6] = 0
ins[19:] = 0

# Fourier expansion of insolation function
y = ins
t = hours
N = len(hours)
i = arange(N)
delta_T = 1
T = N*delta_T
harmonics = zeros((4,N))
coeff = zeros((4,2))
amplitudes = zeros(4)
for k in range(4):
    coeff[k,:] = [(2/N) * sum(y * cos(2*pi*(k+1)*i/N)), (2/N) * sum(y * sin(2*pi*(k+1)*i/N))]
    harmonics[k,:] = coeff[k,0] * cos(2*pi*(k+1)*t/T) + coeff[k,1] * sin(2*pi*(k+1)*t/T)
    amplitudes[k] = sqrt(coeff[k,0]**2 + coeff[k,1]**2)

# plot Fourier expansion of insolation cycle
fig = plt.figure(figsize=(7.5,4.0))
plt.plot(t,ins,c='k',alpha=0.6,label='Insolation function')
plt.plot(t,harmonics[0,:],c='red',
         label='k = 1 (diurnal); C = {0:.03f}°C'.format(amplitudes[0]))
plt.plot(t,harmonics[1,:],c='blue',
         label='k = 2 (semi-diurnal); C = {0:.03f}°C, {1:.0f}% of k = 1'.format(amplitudes[1],
                                                                                100*amplitudes[1]/amplitudes[0]))
plt.plot(t,harmonics[2,:],c='orange',
         label='k = 3 (ter-diurnal); C = {0:.03f}°C, {1:.0f}% of k = 1'.format(amplitudes[2],
                                                                               100*amplitudes[2]/amplitudes[0]))
plt.plot(t,harmonics[3,:],c='green',
         label='k = 4 (quadra-diurnal); C = {0:.03f}°C, {1:.0f}% of k = 1'.format(amplitudes[3],
                                                                                  100*amplitudes[3]/amplitudes[0]))
plt.plot(t,ins.mean() + harmonics.sum(axis=0),c='k',lw=2.5,ls='--',alpha=0.6,label='Sum of k = 1 to 4')
plt.ylim([plt.ylim()[0],plt.ylim()[1]+0.55])
plt.legend(ncol=1,loc='upper left',fontsize=8,frameon=False)
plt.xlim([0,23])
plt.xlabel('Hour of day in local time (GMT+11)')
plt.ylabel('Insolation (no units)')
plt.tight_layout()
plt.savefig(data_dir + 'fig_4_insolation_fourier.pdf')
plt.close()

# prewhiten time series using lag-1 autocorrelation
anomalies = data.values - data.values.mean()
alpha = (anomalies[:-1] @ anomalies[1:]) / (anomalies @ anomalies)
alpha = data.autocorr(lag=1)  # almost equivalent to above
data_prewhitened = data.values[1:] - alpha * data.values[:-1]

# compute power spectra of original and prewhitened series
window = 24*31  # = 744
freq, power = signal.welch(data.values,fs=24,window='hanning',nperseg=window,noverlap=window/2)
freq_pw, power_pw = signal.welch(data_prewhitened,fs=24,window='hanning',nperseg=window,noverlap=window/2)

# plot power spectra
fig = plt.figure(figsize=(7.5,3.5))
plt.semilogy(freq,power,c='k',label='Original series')
plt.plot(freq_pw,power_pw,c='b',label=r'Prewhitened series ($\alpha = $' + '{0:.03f})'.format(alpha))
plt.legend(loc='upper right',frameon=False)
plt.xlabel('Frequency (cycles per day)')
plt.ylabel('Power')
plt.tight_layout()
plt.savefig(data_dir + 'fig_5_power_spectra.pdf')
plt.close()

# compute daily means and prewhiten
daily_means = data.resample('D').mean()
alpha = daily_means.autocorr(lag=1)
data_means_prewhitened = daily_means.values[1:] - alpha * daily_means.values[:-1]

# compute power spectra of original and prewhitened series
window = 365/2
freq, power = signal.welch(daily_means.values,fs=1,window='hanning',nperseg=window,noverlap=window/2)
freq_pw, power_pw = signal.welch(data_means_prewhitened,fs=1,window='hanning',nperseg=window,noverlap=window/2)

# plot power spectra
fig = plt.figure(figsize=(7.5,3.5))
plt.semilogy(freq,power,c='k',label='Original series')
plt.plot(freq_pw,power_pw,c='b',label=r'Prewhitened series ($\alpha = $' + '{0:.03f})'.format(alpha))
plt.legend(loc='upper right',frameon=False)
plt.xlabel('Frequency (cycles per day)')
plt.ylabel('Power')
plt.tight_layout()
plt.savefig(data_dir + 'fig_6_power_spectra_daily_means.pdf')
plt.close()