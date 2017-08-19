import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import shapely
from geopandas import GeoDataFrame
from shapely.geometry import Point
import datetime as dt
import glob
import scipy.stats as st
import time


from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import probplot
from matplotlib import gridspec
from scipy import stats


conf_perc = 0.95

def percent_cb(complete, total):
    sys.stdout.write('.')
    sys.stdout.flush()



crs = {'init':'epsg:4326'}
today_ = dt.date.today()

'''
### Joining with administrative boundaries
Now bring in the administrative boundaries so we can join them in the next step
'''


tracts = glob.glob('admin_boundaries/tl_2016_25_tract/*.shp')[0]
tracts = gpd.read_file(tracts,driver='ESRI Shapefile')
tracts = tracts.to_crs({'init':'epsg:4326'})
tracts = tracts.rename(columns={'GEOID':'ct10_id'})

towns = glob.glob('admin_boundaries/towns/*.shp')[0]
towns = gpd.read_file(towns,driver='ESRI Shapefile')
towns = towns.to_crs({'init':'epsg:4326'})
# towns = towns.rename(columns={'GEOID':'ct10_id'})

zipcodes = glob.glob('admin_boundaries/zipcodes/*.shp')[0]
zipcodes = gpd.read_file(zipcodes,driver='ESRI Shapefile')
zipcodes = zipcodes.to_crs({'init':'epsg:4326'})

neighborhoods = glob.glob('admin_boundaries/neighborhoods/*.shp')[0]
neighborhoods = gpd.read_file(neighborhoods,driver='ESRI Shapefile')
neighborhoods = neighborhoods.to_crs({'init':'epsg:4326'})

col_dict={'neighborhoods':'Neighborho','zipcodes':'objectid','towns':'TOWN_ID','tracts':'ct10_id'}
file_dict={'neighborhoods':neighborhoods,'zipcodes':zipcodes,'towns':towns,'tracts':tracts}



# Selects the latest file by unix date on a directory
def select_date_file(data_path):
    files = sorted([f for f in os.listdir(data_path) if f.endswith(".csv") ])
    filename = sorted(files, key=lambda x: float(float(x[:x.find('_')])))[-1]
    return filename

def data_creation():
	data_path = os.environ['output_path']
	raw_data = os.path.join(data_path, select_date_file(data_path))

	data = pd.read_csv(raw_data)


	data=data.drop(['uniqueid','Unnamed: 0.1','original_title', 'cl_title', 'created_at','updated_at','tract10',
	                'tract10_fwd_geocode', 'census_tract','rev_geolocated', 'COMMUNITY_Merge', 'ADDR_NUM_merge', 
	                'BASE_merge','latitude_merge', 'longitude_merge', 'joint_addresses_merge'],axis=1)
	data = data.rename(columns={'Unnamed: 0': 'unique_id'})

	data['timestamp']=[pd.to_datetime(each) for each in data.post_at]
	data=data[(data.ask>100) & (data.ask<12000)]
	data=data[data['timestamp'].dt.year>2015]

	geom = [Point(xy) for xy in zip(data.latitude,data.longitude)]
	data = GeoDataFrame(data,crs=crs,geometry=geom)

	print("Dimensions of the filtered data are {} by {}".format(data.shape[0],data.shape[1]))

	return data

'''
## First, let's get some stats on the data

We want to see the following:
* Overall size of our filtered data
* Box-plot of the price per month broken down by number of bedrooms
* A log distribution for each of the bedroom tiers 
'''

### First get some stats on the data

### Box plots
def box_plot(data): 
	fig, ax = plt.subplots(figsize=(16,8))

	data_boxplot = []
	for i in np.arange(0,6):
	    data_boxplot.append(data[data.bedrooms ==i].ask.values)


	bp_dict = ax.boxplot(data_boxplot, 0,'r+',vert=False)
	labels = np.arange(0,6)


	# bp_dict=plt.boxplot(data_boxplot,0,'r+')
	for line in bp_dict['medians']:
	    # get position data for median line
	    x, y = line.get_xydata()[1] # top of median line
	    # overlay median value
	    plt.text(x, y, '%.1f' % x,horizontalalignment='center') # draw above, centered
	ax.set_yticklabels(labels)

	ax.set_xscale('log')
	plt.title('Median Rent Distribution by Number of Bedrooms for {}/{}/{}'.format(today_.day,today_.month,today_.year))
	chart_path = os.environ['chart_path'] # 'Data/Output/charts'
	plt.savefig(chart_path+'/{}_ask_boxplot.png'.format(repr(time.time()))))
	print('Boxplot written')
	plt.close()


### Time Series
def time_series_plot(data): 
	fig = plt.figure(figsize=(12,8))

	for i in range(6):
	    ### Group data by year and week 
	    X = data[data.bedrooms ==i]
	    N =X.shape[0]
	    X = X.groupby([X['timestamp'].dt.year,X['timestamp'].dt.week]).agg([np.mean,np.std,'count'], ddof=1)
	    y = [dt.datetime.strptime(' '.join([str(x) for x in (y,m,0)]), "%Y %W %w") for y,m in X.index.values]
	    
	    plt.plot(y,X.ask['mean'], linewidth=2.0,alpha=0.5,label="{} bd, N = {}".format(i,N))

	plt.legend()
	plt.tight_layout()
	plt.title('Median Rent Time Series by Number of Bedrooms for {}/{}/{}'.format(today_.day,today_.month,today_.year))
	chart_path = os.environ['chart_path'] # 'Data/Output/charts'
	plt.savefig(chart_path+'/{}_ask_timeseries.png'.format(repr(time.time()))))
	print('Time Series written')
	plt.close()


def hist_plot(data): 

	fig = plt.figure(figsize=(14,24))

	gs = gridspec.GridSpec(6, 2,width_ratios=[4, 1])

	N = 60
	for i in range(6):
	    ### Create the logged data
	    X = data[data.bedrooms ==i]
	    ln_ask = np.log(X.ask+1)
	    
	    ### ax1: Log probability distribution plot
	    ax1 = fig.add_subplot(gs[i,:])
	    counts, bins, bars =ax1.hist(ln_ask, bins=N,alpha=0.6,edgecolor = 'white',normed=True,facecolor='orange')

	    # Plot the expected PDF of the data
	    mu,std = norm.fit(ln_ask)
	    xmin, xmax = plt.xlim()
	    x = np.linspace(xmin,xmax, N)
	    p = norm.pdf(x, mu, std)
	    ax1.plot(x, p, 'k', linewidth=1,alpha=.4)
	    ax1.set_xlim([6.5,xmax])
	    ### Get the Shapiro-Wilks statistic for normality
	    W,p,a_test = shapiro(counts,reta=True)
	    
	    ### Create the title
	    ax1.set_title('{0}/{1}/{2} | {3} Bedrooms | Fit results: mu = {4:.2f},  std = {5:.2f} | S-W p-value: {6:.5f}'.format(today_.month,today_.day,today_.year,i, mu,std,p))

	    ### Create a QQ plot of the data as a visual normality inspection
	    ax2 = fig.add_subplot(gs[i,1])
	    stats.probplot(ln_ask, dist="norm",plot=ax2)

	plt.tight_layout()
	chart_path = os.environ['chart_path'] # 'Data/Output/charts'
	plt.savefig(chart_path+'/{}_ask_histogram.png'.format(repr(time.time()))))
	plt.close()
	print('Time Series written')



### Join by dataframe
### admin_shp -> pass in the geodataframe 
def admin_agg_spa_group(data,admin_shp):
    data_admin =  gpd.sjoin(data, admin_shp, how="inner", op='intersects')
    return data_admin

### Join by dataframe
### attr_colu -> pass in field to join by
def admin_agg_att_group(data,attr_colu = 'ct10_id'):
    data_admin = data.groupby(attr_colu).mean().reset_index()
    data_admin = tracts.merge(cl_tracts_sum[[attr_colu,'ask']],on=admin_name)
    return data_admin


'''
### Data comparison 
0. Preparation: 
    * Get the logs of the ask prices, as we know that the data are log-normally distributed.
    * Aggregate by the census tract and number of bedrooms.
    * Remove the long right-hand tail as this will skew the data and we consider these "outliers".  Our heuristic
    will be take 95% of the data from the mean.
    * If there are seasonal trends, then we need to subtract that from our ask prices.
1. Take the mean and standard deviations value for each bedroom size and each tract 
2. Create a confidence intervals for each tract by estimating the sample standard deviation.
3. Finally, we want to compare each census tract against a Zillow API number
    * Temp API ID: `X1-ZWz1fwr7hj8f0r_4234r`
    * Use the GetSearchResults API in Zillow to grab the `zpid` (property ID) for representative listings for that tract.  **Let's talk about how to create these representative listings**
    * Using the `zpid` we can then query prices for this property using the GetZestimate API, which gives us the following
        * The most recent property Zestimate
        * The date the Zestimate was computed
        * The valuation range
4. Once we have prices from Zillow for a set of representative points in each tract, we can then compare these means and valuation ranges with our own data. 
5. Seasonality?  Let's look at the seasonality breakdown for each bedroom
'''

### 1. Preparation
### Get the logs of the ask prices, as we know that the data are log-normally distributed.

### Census tracts
def group_data(data,field='ct10_id'):
    data_agg_group = data.groupby([field,'bedrooms']).agg([np.mean,np.std,'count'], ddof=1)
    return data_agg_group


def remove_outliers(data,data_agg,admin_type):
	# ### Census tracts
	data_final = pd.merge(data,data_agg.log_ask,left_on=[col_dict[admin_type],'bedrooms'],right_index=True)
	data_final = data_final[(data_final.log_ask<data_final['mean']+3*data_final['std']) \
	                        &(data_final.log_ask>data_final['mean']-3*data_final['std'])]
	return data_final

def main():
	data= data_creation()
	data['log_ask']=np.log(data['ask'])

	box_plot(data)
	time_series_plot(data)
	hist_plot(data)
	
	for admin_type in col_dict.keys():
		
		if admin_type=='tracts':
			data_byadmin= data
		else:
			data_byadmin= admin_agg_spa_group(data,file_dict[admin_type])
		
		### Change the field based on the level of aggregation you're in
		data_agg = group_data(data_byadmin,col_dict[admin_type])
		# print(data_agg.head())
		print("Data aggregated")
		### Remove the outliers
		data_final = remove_outliers(data_byadmin,data_agg,admin_type)
		print("Outliers removed")
		### 95% confidence
		confidence = float(1-conf_perc)/2
		z_score = st.norm.ppf(1- confidence)
		data_final['CI_L'] = data_final['mean']-z_score*data_final['std']
		data_final['CI_H'] = data_final['mean']+z_score*data_final['std']
		print("Confidence intervals created")
		analysis_path = os.environ['analysis_path'] # 'Data/Output/analysis'
		data_final.to_csv(analysis_path+'/{}_{}_output.csv'.format(repr(time.time())),admin_type))
	

if __name__ == '__main__':
	main()
