from numpy import *
import numpy.ma as ma
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.basemap import Basemap

import pandas.plotting._converter as pandacnv   # only necessary due to Pandas 0.21.0 bug with Datetime plotting
pandacnv.register()
set_printoptions(threshold=100)   # faster

# filepaths
data_dir = '/Users/Ethan/Documents/Utility/Matlab/ATMS552/HW3/'
fig_dir = '/Users/Ethan/Documents/UW/By course/2018-01 - ATM S 552 (Dennis L. Hartmann)/' \
          '2018-02-21 - assignment 3 - specifications and results/'

# load SST data (from https://www.esrl.noaa.gov/psd/data/gridded/data.noaa.ersst.html)
data = xr.open_dataset(data_dir + 'sst.mnmean.v4.nc')

# subset to period of interest; only include full years
data = data.sel(time=slice('1900-01','2017-12'))

# check that land is already masked with NaNs (it is)
# plt.imshow(data['sst'].sel(time='2017-12')[0])

# remove time- and global-mean
data['sst'] = data['sst'] - data['sst'].mean(dim='time')

# construct, then remove seasonal cycle
climo = data['sst'].groupby('time.month').mean(axis=0)
for t_idx, time in enumerate(pd.DatetimeIndex(data['time'].values)):
    data['sst'].loc[time] -= climo.sel(month=time.month)

# remove long-term trend by subtracting global mean SST from each time step
lat_weights = cos(data['lat']*pi/180)  # weight average by grid cell area, which varies with latitude
global_means \
    = pd.Series(index=pd.DatetimeIndex(data['time'].values),
                data=[mean(ma.average(ma.masked_invalid(data['sst'].sel(time=time)),axis=0,weights=lat_weights))
                      for time in data['time'].values])
for t_idx, time in enumerate(data['time'].values):
    data['sst'].loc[time] -= global_means[time]
    
# plot long-term global trend
plt.figure(figsize=(7,3))
plt.plot(global_means,c='k')
plt.xlim([min(global_means.index),max(global_means.index)])
years = mdates.YearLocator(10)
plt.gca().xaxis.set_major_locator(years)
plt.title('Global SST trend')
plt.ylabel('SST (°C)')
plt.tight_layout()
plt.savefig(fig_dir + 'fig_1_global_trend.pdf')
plt.close()

# weight SST anomalies by grid cell size (sqrt necessary because units of X^2 [variance])
cell_weights = tile(sqrt(lat_weights),(len(data['lon']),1)).T
data['sst'] *= cell_weights

# areas of interest
original_data = data.copy()
area_strs = ['global','southern_ocean']
area_lat_lons = [[[90,-90],[0,360]],[[-45,-85],[0,360]]]
area_levels = [arange(-0.1,0.105,0.01),arange(-0.24,0.245,0.02)]

for a_idx, area_str in enumerate(area_strs):
    # subset to desired area
    data = original_data.sel(lat=slice(*area_lat_lons[a_idx][0]),lon=slice(*area_lat_lons[a_idx][1]))

    # flatten spatial dimension and remove NaNs
    # note: to reverse, fill NaN array of shape (N_lon*N_lat,N_time) at spatial indices given by true_idx_vector,
    #                   then reshape to (N_time,N_lat,N_lon) using C-order, then transpose
    N_time = len(data['time'])
    N_lat = len(data['lat'])
    N_lon = len(data['lon'])
    sst = data['sst'].values  # from xarray to naive NumPy array
    sst = sst.T.reshape((N_lon*N_lat,N_time),order='C')
    mask_vector = isnan(sst.mean(axis=1))
    true_idx_vector = where(~mask_vector)
    sst = sst[~mask_vector,:]
    N_data_cells = len(sst)
    lat_grid, lon_grid = meshgrid(data['lat'],data['lon'],indexing='ij')
    
    # estimate autocorrelation of data (necessary for North Test)
    var = diag(sst @ sst.T) / N_time                              # variance of each cell
    autocov = diag(sst[:,:N_time-1] @ sst[:,1:].T) / (N_time-1)   # one-lag covariance (autocovariance)
    autocor = autocov / var                                       # autocorrelation
    autocor_mean = nanmean(autocov) / nanmean(var)                # average autocorrelation
    dof = N_time * (1 - autocor_mean**2) / (1 + autocor_mean**2)  # corrected degrees of freedom
    north_factor = sqrt(2/dof)                                    # standard error of eigenvalues from North et al.
    
    # SVD!
    U, S, V = linalg.svd(sst)                  # S is vector of singular values; no need to apply diag(S)
    eigen = S**2 / N_data_cells                # S is in amplitude (std dev) units; square S to get X^2 (variance) units;
                                               #      divide by N to get eigenvalues
    eigen_var = 100 * eigen / sum(eigen)       # percent of variance explained by each eigenvalue
    eigen_std_err = eigen_var * north_factor   # standard error in eigenvalues (units of percent of variance explained)
    
    # plot eigenvalue spectrum
    # note: first mode explains slightly less of total variance when SVD is done on data with NaNs filled as zeros
    #       rather than truncated entirely from the spatial domain
    N_eigen_to_plot = 12
    plt.figure(figsize=(4.5,4))
    plt.errorbar(range(N_eigen_to_plot),eigen_var[:N_eigen_to_plot],yerr=eigen_std_err[:N_eigen_to_plot],
                 c='k',capsize=5)
    plt.errorbar(0,NaN,yerr=NaN,c='k',fmt='none',capsize=5,label='±1 standard error')
    plt.legend(frameon=False)
    plt.title('Eigenvalue spectrum (first {0} modes only)'.format(N_eigen_to_plot))
    plt.xlabel('Mode number')
    plt.ylabel('Percent of variance explained')
    plt.tight_layout()
    plt.savefig(fig_dir + 'fig_2_{0}_eigenvalue_spectrum.pdf'.format(area_str))
    plt.close()
    
    # eigenmode regressions
    N_modes = 5
    mode_sign = [-1,1,1,1,1]
    for im in range(N_modes):
        pc = V.T[:,im]                        # principal component time series
        pc = (pc - mean(pc)) / std(pc)        # standardize to unit variance (other option: divide by sqrt(eigenvalue)?)
        pc *= mode_sign[im]                   # convert to desired sign
        pc_series = pd.Series(index=data['time'].values,data=pc)
    
        plt.figure(figsize=(7,3))
        plt.plot(pc_series,c='k',lw=0.5)
        plt.plot(pc_series.rolling(window=12*5,center=True).mean(),c='b',lw=1,label='5-year centered rolling mean')
        plt.xlim([min(global_means.index),max(global_means.index)])
        plt.ylim([plt.ylim()[0],plt.ylim()[1]+1.0])
        plt.legend(frameon=False,loc='upper right')
        plt.gca().xaxis.set_major_locator(years)
        plt.title('Principal component #{0}'.format(im+1))
        plt.ylabel('Principal component amplitude')
        plt.tight_layout()
        plt.savefig(fig_dir + 'fig_3_{0}_pc_{1}_amplitude.pdf'.format(area_str,im+1))
        plt.close()
    
        regr = sst @ pc.T / N_data_cells          # regress PC against data to show pattern with amplitude in data units
        regr_all_cells = full((N_lon*N_lat),nan)  # restore NaN cells
        regr_all_cells[true_idx_vector] = regr    # "
        regr_all_cells = regr_all_cells.reshape((N_lon,N_lat),order='C').T  # restore 2D shape
    
        plt.figure(figsize=(7,4))
        if a_idx == 0: m = Basemap(projection='robin',lon_0=0,resolution='c')
        if a_idx == 1: m = Basemap(projection='spstere',boundinglat=-47,lon_0=180,round=True,resolution='i')
        m.contourf(lon_grid,lat_grid,regr_all_cells,area_levels[a_idx],latlon=True,cmap='RdBu_r',extend='both')
        m.drawcoastlines()
        m.fillcontinents(color='k',lake_color='w')
        m.drawparallels(arange(-90.,120.,30.))
        m.drawmeridians(arange(0.,360.,60.))
        m.drawmapboundary(fill_color='w')
        plt.colorbar(orientation='vertical',extend='both',shrink=0.8,label='°C')
        plt.title('Regression of principal component #{0} on SST'.format(im+1))
        plt.savefig(fig_dir + 'fig_4_{0}_pc_{1}_regression.pdf'.format(area_str,im+1))
        plt.close()