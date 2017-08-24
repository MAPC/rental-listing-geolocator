# coding: utf-8
import pandas as pd
import numpy as np
import os
import re
import requests
import json
from scipy.spatial import cKDTree as KDTree
import time
import subprocess


# The function opens all the CSV files in a directory, and joins them into a large DF
def open_all_csv(path):
    dfs = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            curr_df = pd.read_csv(os.path.join(path, file))
            dfs.append(curr_df)
    return pd.concat(dfs, ignore_index=True)


# Perhaps this needs to be rewritten for S3
# Breaks all the data into smaller chunks to process
def break_data(data, chunk_size, path):
    for g, df in data.groupby(np.arange(len(data)) // chunk_size):
        df.to_csv(path+str(g)+'.csv')


# Opens a set of CSVs based on a list of town names of interest
def open_selected_csv(path, towns):
    dfs = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            if file[:-4].lower().capitalize() in towns:
                try:
                    curr_df = pd.read_csv(os.path.join(path, file))[['census_tract','latitude','longitude','ADDRESS_ID','ADDR_NUM','STREETNAME','UNIT','ZIPCODE','COMMUNITY','TOWN','NUM1','NUM1_TXT','NUM1_SFX','NUM2_PFX','BASE','POST_TYPE','USPS_TYPE','STRNAME_ID']]
                except: 
                    curr_df = pd.read_csv(os.path.join(path, file))[['census_tract','latitude','longitude','FULL_ADDRE','UNIT','STREETNAME','BASE','STREET_SUF','USPS_TYPE','ADDR_NUM','COMMUNITY','ZIPCODE','PARCEL']]
                dfs.append(curr_df)
    return pd.concat(dfs, ignore_index=True)


# Opens a set of Excel or CSV files based on a list of town names of interest
def open_selected_xls(path, towns):
    dfs = []
    for file in os.listdir(path):
        curr_df = None
        if file.endswith(".xlsx"):
            if file[:-5].lower().capitalize() in towns:
                curr_df = pd.read_excel(os.path.join(path, file))
                dfs.append(curr_df)
        elif file.endswith('.csv'):
            if file[:-4].lower().capitalize() in towns:
                curr_df = pd.read_csv(os.path.join(path, file))
                dfs.append(curr_df)
    return pd.concat(dfs, ignore_index=True)


# Converts numbers into strings
def nums_to_str(x):
    try: return str(int(x))
    except: return str(x)


# Eliminates the street types from a string with regex
def street_re(street_text):
    r1 = '^.*(?=((?i)street|st|road|rd|avenue|ave))'
    try:
        regex = re.search(r1, street_text)
        return regex.group(0)
    except: 
        return ''


# Looks for a street name in the listing with regex
def strip_address(x, street_names_merge):
    for i, row in street_names_merge.iteritems():
        try:
            r1 = '(\d+\s+)(?i)'+ row.strip() + '.*'#'\w+'
            regex = re.search(r1, x)
            return regex.group(0)
        except: 
            pass


# Looks for a street type with regez
def strip_st_type(x, regex_st_types):
    try:
        r1 = '.*(?=(?i)'+regex_st_types+')'
        r1 = regex_st_types
        regex = re.search(r1, x)
        return regex.group(0).strip().lower()
    except: 
        pass


# Processes a DF, and applies the regex functions on the columns to find street names in the listing title
def worker(path, street_names_merge, strip_address, strip_st_type, regex_st_types, output_path):
    current_df = pd.read_csv(path, index_col=0)
    
    pad_mapper_addr = current_df.title.apply(strip_address, args=(street_names_merge,))
    pad_mapper_addr = pad_mapper_addr.apply(strip_st_type, args=(regex_st_types,))
    print 'Processed'
    pad_mapper_addr.to_csv(output_path)
    return


# Open a set of CSVs and creates a big DF
def open_series_csv(path):
    dfs = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            dfs.append(pd.Series.from_csv(os.path.join(path, file), parse_dates=False))
    return pd.concat(dfs)


# Function that turns integers into strings and keeps NAs
def val_to_agg(val):
    if pd.isnull(val):
        return val
    else: 
        return str(val)


# Function that queries the MApzen API to find the addresses and their lat/lon values for a given lat/lon
def mapzen_api(mapzen_df, index):
    api_key = MAPC_mapzen

    lat = mapzen_pad_mapper.loc[index].latitude
    lon = mapzen_pad_mapper.loc[index].longitude
    url = 'https://search.mapzen.com/v1/reverse?point.lat='+str(lon)+'&point.lon='+str(lat)+'&size=3&api_key=' + api_key

    # GET
    r = requests.get(url).text
    
    address_dict = json.loads(r)['features'][0]
    if 'housenumber' in address_dict['properties']:
        addr_num = address_dict['properties']['housenumber']
    else: addr_num = None 

    if 'street' in address_dict['properties']:
        base = address_dict['properties']['street']
    else: base = None 

    if 'locality' in address_dict['properties']:
        community = address_dict['properties']['locality']
    else: community = None 

    if 'label' in address_dict['properties']:
        joint_addresses = address_dict['properties']['label']
    else: joint_addresses = None 

    if 'confidence' in address_dict['properties']:
        confidence = address_dict['properties']['confidence']
    else: accuracy = None 


    mapzen_df.loc[index] = [addr_num, base, community, address_dict['geometry']['coordinates'][0],
                           address_dict['geometry']['coordinates'][1], joint_addresses, confidence]


# Deletes all the files in a folder
def delete_folder_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def mapzen_api_keys(mapzen_df, keys):
    indices = mapzen_df.index.sort_values()
    api_key = MAPC_mapzen

    lat = keys[0]
    lon = keys[1]
    url = 'https://search.mapzen.com/v1/reverse?point.lat='+str(lon)+'&point.lon='+str(lat)+'&size=3&api_key=' + api_key

    # GET
    r = requests.get(url).text
    address_dict = json.loads(r)['features'][0]
    if 'housenumber' in address_dict['properties']:
        addr_num = address_dict['properties']['housenumber']
    else: addr_num = None 

    if 'street' in address_dict['properties']:
        base = address_dict['properties']['street']
    else: base = None 

    if 'locality' in address_dict['properties']:
        community = address_dict['properties']['locality']
    else: community = None 

    if 'label' in address_dict['properties']:
        joint_addresses = address_dict['properties']['label']
    else: joint_addresses = None 

    if 'confidence' in address_dict['properties']:
        confidence = address_dict['properties']['confidence']
    else: accuracy = None 

    if len(indices) == 0:
        curr_index = 0
    else: 
        curr_index = indices[-1] + 1

    mapzen_df.loc[curr_index] = [addr_num, base, community, address_dict['geometry']['coordinates'][0],
                           address_dict['geometry']['coordinates'][1], joint_addresses, confidence, lat, lon]
    return True

# Selects the latest file by unix date on a directory
def select_date_file(data_path):
    files = sorted([f for f in os.listdir(data_path) if f.endswith(".csv") ])
    filename = sorted(files, key=lambda x: float(float(x[:x.find('_')])))[-1]
    return filename

# ENV VARS
data_path = os.environ['data_path'] #'Data'
worker_folder = os.environ['worker_folder'] #'Data/worker_data/'
mapc_path = os.environ['mapc_path'] # 'CSV/mapc/'
point_path = os.environ['point_path'] # 'Points'
csv_path = os.environ['csv_path'] # 'CSV'
worker_proc = os.environ['worker_proc'] # 'Data/worker_processed/'
output_path = os.environ['output_path'] # 'Data/Output'
MAPC_mapzen = os.environ['MAPC_mapzen'] # 


# First we import the clean dataset into a Pandas DF
rental_df = pd.read_csv(os.path.join(data_path, select_date_file(data_path)), index_col=0)

# We subset the dataset to get only the padmapper listings
pad_mapper = rental_df.loc[(rental_df['source_id'] == 2) & (rental_df['longitude']!=0)]

# Delete the contents of the directory
delete_folder_files(worker_folder)

# Break the data into chunks of 100 listings
chunk_size = 100
break_data(pad_mapper, chunk_size, worker_folder)

# Get the names of all the towns within the MAPC
mapc_towns = list(pd.read_csv(os.path.join(mapc_path, 'MAPC Towns.csv')).municipal)


# Get the geolocations associated with the MAPC
geolocations = open_selected_csv(point_path, mapc_towns)


# Set the index of the geolocations as a column
geolocations['indices'] = geolocations.index

# Create a DF with all the street types 
street_types = pd.concat([geolocations.STREET_SUF, geolocations.USPS_TYPE, geolocations.POST_TYPE])
street_types = street_types.dropna().drop_duplicates().str.lower()

# Cosntruct a large string with all the street types
regex_st_types = ''
for i in street_types:
    regex_st_types +=  i +'|'
regex_st_types = regex_st_types[:-1]
regex_st_types = '.*(?=(?i)'+ regex_st_types +')'


# Join the street names and the address number into a single column
geolocations['joint_addresses'] = geolocations.ADDR_NUM.apply(nums_to_str) + ' ' + geolocations.BASE
geolocations['joint_addresses'] = geolocations['joint_addresses'].str.lower()
cleaned_geolocations = geolocations.drop_duplicates(subset=['joint_addresses', 'COMMUNITY'])

# # This only happens if new street names are added, if not just read the local copy
# # # Locally saved street names
# # cleaned_streets = pd.Series.from_csv('CSV/streetnames.csv', index_col=0)

# # # Get all the street names in the MAPC region
# # street_name_path = 'GeoCode/MA'
# # street_name_dfs = open_selected_xls(street_name_path, mapc_towns)


# # # Merge both street formats into a DF to get all the street names
# # base_names = street_name_dfs.BASE.dropna().drop_duplicates()
# # full_street = street_name_dfs.FULL_STREET_NAME.dropna().drop_duplicates()
# # street_names_merge = pd.concat([base_names, full_street, cleaned_streets]).str.lower()
# # street_names_merge = street_names_merge.drop_duplicates()

# Local Copy of all the street names
street_names_merge = pd.Series.from_csv(os.path.join(csv_path, 'streetnames_merge.csv'), index_col=0)


# Delete the contents of the directory
delete_folder_files(worker_proc)

# get the number of files in the directory
num_of_chunks = [file for file in os.listdir(worker_folder) if file.endswith(".csv") ]
# loop through all the files to execute the regex functions
for i in range(len(num_of_chunks)):
    try:
        path = worker_folder + str(i) + '.csv'
        output = worker_proc + str(i) + '.csv'

        print output
        
        worker(path, street_names_merge, strip_address, strip_st_type, regex_st_types, output)
    except Exception as e:
        print(e)

# Opens all the processed files, and creaes a larger series
pad_mapper_addr = open_series_csv(worker_proc)

# Turns the Series into a DF, and joins them with the lat, lon, and census tract
pad_mapper_addr_df = pad_mapper_addr.to_frame(name='joint_addresses').join(pad_mapper[['latitude','longitude','tract10']])

# Rename some columns
pad_mapper_addr_df.rename(columns={'latitude': 'lat_crawled', 'longitude': 'lon_crawled'}, inplace=True)


# Look for the matching addresses in the maser address Db
forward_geocode = pd.merge(cleaned_geolocations, pad_mapper_addr_df, on='joint_addresses', how='inner', left_index=True)
forward_geocode = forward_geocode.drop_duplicates()[['ADDR_NUM', 'BASE', 'COMMUNITY', 'latitude', 'longitude', 'joint_addresses','tract10','census_tract', 'lat_crawled', 'lon_crawled']]


# Make sure that the merged names match the census tract number
forward_geocode_nodups = forward_geocode[(forward_geocode['tract10'] == forward_geocode['census_tract'])]
forward_geocode_nodups = forward_geocode_nodups[['ADDR_NUM', 'BASE', 'COMMUNITY', 'latitude', 'longitude', 'joint_addresses', 'tract10','census_tract']]


# Add a flag for the listings that were successfully geolocated
pad_mapper['fwd_geolocated'] = False
pad_mapper.loc[forward_geocode_nodups.index, 'fwd_geolocated'] = True

# Join the newly geolocated addresses 
pad_mapper = pad_mapper.join(forward_geocode_nodups, rsuffix='_fwd_geocode')
pad_mapper = pad_mapper.loc[:,~pad_mapper.columns.duplicated()]


# Create a KD Tree to query for closest neighbor to every address in the DB
# for each row in df2, we want to join the nearest row in df1
# based on the column "d"
not_geocoded = pad_mapper.loc[pad_mapper['fwd_geolocated'] == False]
join_cols = ['latitude', 'longitude']

# We create the KD Tree with the Point geolocations
tree = KDTree(geolocations[join_cols])
# We query the KD Tree with the listing not geocoded
distance, indices = tree.query(not_geocoded[join_cols])


# The decimal degrees have to be transformed to meters
d = {'distance': distance * 111325, 'indices': indices}
# Create a DF with the dictionary we just created
dist_df = pd.DataFrame(data=d, index=not_geocoded.index)
# Only grab the entries that are within 20 mts
dist_df.loc[dist_df['distance'] > 20, 'indices'] = None
dist_df = dist_df.dropna()
dist_df['indices'] = dist_df['indices'].astype('int')


# Merge with the master address dataset
reverse_geocode = geolocations.iloc[dist_df['indices']]
# Set the index to the original dataset index
reverse_geocode = reverse_geocode.set_index(dist_df.index)[['ADDR_NUM', 'BASE', 'COMMUNITY', 'latitude', 'longitude', 'joint_addresses']]


# Set the reverse geolocsated flag to False by default
pad_mapper['rev_geolocated'] = False
# Assign the flag to True to the listings that have been geolocated
pad_mapper.loc[reverse_geocode.index, 'rev_geolocated'] = True
# Join the spatial information to all the listings
pad_mapper = pad_mapper.join(reverse_geocode, rsuffix='_rev_geocode')

# Remove duplicates
pad_mapper = pad_mapper[~pad_mapper.index.duplicated(keep='first')]
pad_mapper = pad_mapper.loc[:,~pad_mapper.columns.duplicated()]


# Transform the columns that have numbers into strings to merge multiple columns into a single column
pad_mapper['latitude_fwd_geocode'] = pad_mapper['latitude_fwd_geocode'].apply(val_to_agg)
pad_mapper['latitude_rev_geocode'] = pad_mapper['latitude_rev_geocode'].apply(val_to_agg)
pad_mapper['longitude_fwd_geocode'] = pad_mapper['longitude_fwd_geocode'].apply(val_to_agg)
pad_mapper['longitude_rev_geocode'] = pad_mapper['longitude_rev_geocode'].apply(val_to_agg)
pad_mapper['ADDR_NUM'] = pad_mapper['ADDR_NUM'].apply(val_to_agg)
pad_mapper['ADDR_NUM_rev_geocode'] = pad_mapper['ADDR_NUM_rev_geocode'].apply(val_to_agg)


# Select a set of columns to drop from the overall dataset
drop_columns = ['COMMUNITY_rev_geocode','COMMUNITY', 
                'ADDR_NUM_rev_geocode','ADDR_NUM', 
                'BASE_rev_geocode','BASE',
                'latitude_rev_geocode','latitude_fwd_geocode',
                'longitude_fwd_geocode','longitude_rev_geocode',
                'joint_addresses_rev_geocode','joint_addresses']

# Merge the forward and backward geolocation columns into a single one
pad_mapper['COMMUNITY_Merge'] = pad_mapper[['COMMUNITY', 'COMMUNITY_rev_geocode']].fillna('').sum(axis=1)
pad_mapper['ADDR_NUM_merge'] = pad_mapper[['ADDR_NUM', 'ADDR_NUM_rev_geocode',]].fillna('').sum(axis=1) ####
pad_mapper['BASE_merge'] = pad_mapper[['BASE', 'BASE_rev_geocode']].fillna('').sum(axis=1)
pad_mapper['latitude_merge'] = pad_mapper[['latitude_fwd_geocode', 'latitude_rev_geocode']].fillna('').sum(axis=1) ####
pad_mapper['longitude_merge'] = pad_mapper[['longitude_fwd_geocode', 'longitude_rev_geocode']].fillna('').sum(axis=1) #####
pad_mapper['joint_addresses_merge'] = pad_mapper[['joint_addresses', 'joint_addresses_rev_geocode']].fillna('').sum(axis=1)


# Drop the repeated columns
pad_mapper_clean = pad_mapper.drop(drop_columns, axis=1)
# Obtain the listings that were not forward or backward geolocated
mapzen_pad_mapper = pad_mapper_clean[(pad_mapper_clean.fwd_geolocated == False)&(pad_mapper_clean.rev_geolocated==False)]
# print mapzen_pad_mapper
# Create an empty dataframe to hold the values to be queried from the Mapzen API
mapzen_df = pd.DataFrame(columns=['ADDR_NUM_mapozen', 'BASE_mapzen', 'COMMUNITY_mapzen', 'latitude', 'longitude', 'joint_addresses_mapzen','mapzen_confidence'])

# Iterate through the indices that are left, and geocode them with the mapzen API
for index in mapzen_pad_mapper.index:
    # print index
    try:
        mapzen_api(mapzen_df, index)
    except Exception as e:
        print(e)

# Add flags for the listings that were geocoded with Mapzen
pad_mapper_clean['mapzen_geolocated'] = False
pad_mapper_clean.loc[mapzen_df.index, 'mapzen_geolocated'] = True
# Join the listing data with the Mapzen geoded entries
pad_mapper_clean = pad_mapper_clean.join(mapzen_df, rsuffix='_mapzen_geocode')
pad_mapper_clean = pad_mapper_clean[~pad_mapper_clean.index.duplicated(keep='first')]


# Transform numeric values into strings for aggregation
pad_mapper_clean['latitude_mapzen_geocode'] = pad_mapper_clean['latitude_mapzen_geocode'].apply(val_to_agg)
pad_mapper_clean['longitude_mapzen_geocode'] = pad_mapper_clean['longitude_mapzen_geocode'].apply(val_to_agg)
pad_mapper_clean['ADDR_NUM_mapozen'] = pad_mapper_clean['ADDR_NUM_mapozen'].apply(val_to_agg)


# Define columns to be dropped later
drop_mapzen = ['latitude_mapzen_geocode','longitude_mapzen_geocode', 'ADDR_NUM_mapozen',
              'BASE_mapzen', 'COMMUNITY_mapzen', 'joint_addresses_mapzen']

# Merge the mapzen columns with the forward and backward geolocated values
pad_mapper_clean['latitude_merge'] = pad_mapper_clean[['latitude_merge', 'latitude_mapzen_geocode']].fillna('').sum(axis=1) 
pad_mapper_clean['longitude_merge'] = pad_mapper_clean[['longitude_merge', 'longitude_mapzen_geocode']].fillna('').sum(axis=1) 
pad_mapper_clean['ADDR_NUM_merge'] = pad_mapper_clean[['ADDR_NUM_merge', 'ADDR_NUM_mapozen']].fillna('').sum(axis=1) 
pad_mapper_clean['BASE_merge'] = pad_mapper_clean[['BASE_merge', 'BASE_mapzen']].fillna('').sum(axis=1) 
pad_mapper_clean['COMMUNITY_Merge'] = pad_mapper_clean[['COMMUNITY_Merge', 'COMMUNITY_mapzen']].fillna('').sum(axis=1) 
pad_mapper_clean['joint_addresses_merge'] = pad_mapper_clean[['joint_addresses_merge', 'joint_addresses_mapzen']].fillna('').sum(axis=1) ####


# Drop repeated columns
pad_mapper_clean = pad_mapper_clean.drop(drop_mapzen, axis=1)


#################################################
# Here we start processing the Craigslist listings
craigslist = rental_df.loc[(rental_df['source_id'] == 1) & (rental_df['longitude']!=0)]

# We group the listings by lat and lon to then tag the ones that are over repeated
craigs_grouped = craigslist.groupby(['latitude', 'longitude'])
# We set a threshold of 200 hits of the same lat/lon
filtered_indices = craigs_grouped.filter(lambda x: len(x)>200).index

# Tag the repeated listings
craigslist['repeated_location'] = False
craigslist.loc[filtered_indices, 'repeated_location'] = True


# We use the same tree previously generated and query for the nearest neighbors
distance, indices = tree.query(craigslist[join_cols])

# Here we create a new DF and find the listings closer than 20 mts to the address points
d = {'distance': distance * 111325, 'indices': indices}
dist_df = pd.DataFrame(data=d, index=craigslist.index)
# dist_df['indices'] = dist_df['indices'].astype('int')
dist_df.loc[dist_df['distance'] > 20, 'indices'] = None
dist_df = dist_df.dropna()
dist_df['indices'] = dist_df['indices'].astype('int')

# we create a new DF with the results of the geocoding process 
reverse_geocode = geolocations.iloc[dist_df['indices']]
reverse_geocode = reverse_geocode.set_index(dist_df.index)[['ADDR_NUM', 'BASE', 'COMMUNITY', 'latitude', 'longitude', 'joint_addresses']]

# we join the new reverse geocoded df with the larger DF
craigslist['rev_geolocated'] = False
craigslist.loc[reverse_geocode.index, 'rev_geolocated'] = True
craigslist = craigslist.join(reverse_geocode, rsuffix='_rev_geocode')

# We eliminate duplicate entries 
craigslist = craigslist[~craigslist.index.duplicated(keep='first')]
craigslist = craigslist.loc[:,~craigslist.columns.duplicated()]

# The remaining entries will be geocoded through mapzen
mapzen_craigslist = craigslist[craigslist.rev_geolocated==False]

# We group all the listings by lat lon to reduce the number of calls to the API
mapzen_grouped = mapzen_craigslist.groupby(['latitude', 'longitude'])
filtered_indices_mz = mapzen_grouped.filter(lambda x: len(x)==1).index

# We get the lat lon values of the groups
grouped_keys = mapzen_grouped.groups.keys()


# We create an empty DF to hold the query results
mapzen_df_craigslist = pd.DataFrame(columns=['ADDR_NUM', 'BASE', 'COMMUNITY', 'latitude_mapzen', 'longitude_mapzen', 'joint_addresses','mapzen_confidence', 'latitude', 'longitude'])

# # We query the Mapzen API
for key in grouped_keys:
    # print key
    try: mapzen_api_keys(mapzen_df_craigslist, key)
    except Exception as e:
        print(e)

# We set the index as a md index based on the lat lon
mapzen_multi_index = mapzen_df_craigslist.set_index(['latitude', 'longitude'], drop = False, inplace=False)

# We add the original lat lon as a column, but first we parse them
# lat_lon = [[],[]]
# for i in grouped_keys:
#     lat_lon[0].append(i[0])
#     lat_lon[1].append(i[1])

# mapzen_multi_index_rename = mapzen_multi_index.rename(columns={'latitude': 'latitude_mapzen', 'longitude': 'longitude_mapzen'})

# mapzen_multi_index_rename['latitude'] = lat_lon[0]
# mapzen_multi_index_rename['longitude'] = lat_lon[1]

# We merge the table with non duplicate lat lon values with the original craigslist-mapzen dataset
mapzen_craigslist_merge = pd.merge(mapzen_craigslist, mapzen_multi_index, how='left', on=['latitude', 'longitude'])

# We re-assign the indices
mapzen_craigslist_merge.index = mapzen_craigslist.index
mapzen_craigslist_clean = mapzen_craigslist_merge[['ADDR_NUM_y', 'BASE_y', 'COMMUNITY_y', 'latitude_mapzen', 'longitude_mapzen', 'joint_addresses_y', 'mapzen_confidence']]

# We flag the new geocoded listings as geolocated by mapzen
craigslist['mapzen_geolocated'] = False
craigslist.loc[mapzen_craigslist_clean.index, 'mapzen_geolocated'] = True
craigslist_map = craigslist.join(mapzen_craigslist_clean, rsuffix='_mapzen_geocode')

# Remove duplicates
craigslist_map = craigslist_map[~craigslist.index.duplicated(keep='first')]
craigslist_map = craigslist_map.loc[:,~craigslist_map.columns.duplicated()]

# We translate numeric values to stings to easily merge columns
craigslist_map['latitude_rev_geocode'] = craigslist_map['latitude_rev_geocode'].apply(val_to_agg)
craigslist_map['longitude_rev_geocode'] = craigslist_map['longitude_rev_geocode'].apply(val_to_agg)
craigslist_map['ADDR_NUM'] = craigslist_map['ADDR_NUM'].apply(val_to_agg)

craigslist_map['latitude_mapzen'] = craigslist_map['latitude_mapzen'].apply(val_to_agg)
craigslist_map['longitude_mapzen'] = craigslist_map['longitude_mapzen'].apply(val_to_agg)
craigslist_map['ADDR_NUM_y'] = craigslist_map['ADDR_NUM_y'].apply(val_to_agg)

# We merge the reverse geolocated and mapzen geolocated columns
craigslist_map['COMMUNITY_Merge'] = craigslist_map[['COMMUNITY', 'COMMUNITY_y']].fillna('').sum(axis=1)
craigslist_map['ADDR_NUM_merge'] = craigslist_map[['ADDR_NUM', 'ADDR_NUM_y',]].fillna('').sum(axis=1) ####
craigslist_map['BASE_merge'] = craigslist_map[['BASE', 'BASE_y']].fillna('').sum(axis=1)
craigslist_map['latitude_merge'] = craigslist_map[['latitude_mapzen', 'latitude_rev_geocode']].fillna('').sum(axis=1) ####
craigslist_map['longitude_merge'] = craigslist_map[['longitude_mapzen', 'longitude_rev_geocode']].fillna('').sum(axis=1) #####
craigslist_map['joint_addresses_merge'] = craigslist_map[['joint_addresses', 'joint_addresses_y']].fillna('').sum(axis=1)

# We eliminate extra columns 
drop_columns = ['COMMUNITY_y','COMMUNITY', 
                'ADDR_NUM_y','ADDR_NUM', 
                'BASE_y','BASE',
                'latitude_rev_geocode','latitude_mapzen',
                'longitude_rev_geocode','longitude_mapzen',
                'joint_addresses_y','joint_addresses']
craigslist_clean = craigslist_map.drop(drop_columns, axis=1)

# Merge padmapper and craigslist processed listings 
processed_listings = pd.concat([pad_mapper_clean, craigslist_clean])

#################============================#######################################
# We finally output it ... 
processed_listings.to_csv(os.path.join(output_path, repr(time.time()))+'_processed_listings.csv')


os.system('python data_analysis.py')


