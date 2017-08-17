# coding: utf-8
import pandas as pd
import math
import numpy as np
import os
import re
import requests
import json
from scipy.spatial import cKDTree as KDTree
import multiprocessing
import requests
import boto3
import botocore
import os, shutil


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
    api_key = os.environ['MAPC_mapzen']

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
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

# Function that queries the MApzen API to find the addresses and their lat/lon values for a given lat/lon
def mapzen_api_repeated(mapzen_df, index):
    api_key = os.environ['MAPC_mapozen']

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
                           address_dict['geometry']['coordinates'][1], joint_addresses, confidence, lat, lon]


# filelist = [ f for f in os.listdir(".") if f.endswith(".bak") ]
# for f in filelist:
#     os.remove(f)

s3 = boto3.resource('s3',
            aws_access_key_id=os.environ['MAPC_access'],
            aws_secret_access_key=os.environ['MAPC_secret'])
mapc_bucket = s3.Bucket('rental-listings-data-input')


for object in mapc_bucket.objects.all():
    filename = object.key
    try:
        with open(object.key, 'wb') as data:
            mapc_bucket.download_fileobj(object.key, data)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

# # First we import the clean dataset into a Pandas DF
rental_df = pd.read_csv(filename, index_col=0)

# We subset the dataset to get only the padmapper listings
pad_mapper = rental_df.loc[(rental_df['source_id'] == 2) & (rental_df['longitude']!=0)]


# Delete the contents of the directory
folder = 'Data/worker_data'
delete_folder_files(folder)

# Break the data into chunks of 100 listings
chunk_size = 100
worker_path = 'Data/worker_data/'
break_data(pad_mapper, chunk_size, worker_path)

# Get the names of all the towns within the MAPC
mapc_path = 'CSV/mapc/MAPC Towns.csv'
mapc_towns = list(pd.read_csv(mapc_path).municipal)


# Get the geolocations associated with the MAPC
csv_path = 'Points'
geolocations = open_selected_csv(csv_path, mapc_towns)


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
street_names_merge = pd.Series.from_csv('CSV/streetnames_merge.csv', index_col=0)


# Paths for the files to be read, and the files to be processed
worker_proc = 'Data/worker_processed/'
worker_path = 'Data/worker_data/'

# Delete the contents of the directory
delete_folder_files(worker_proc)

# get the number of files in the directory
num_of_chunks = [file for file in os.listdir(worker_path) if file.endswith(".csv") ]
# loop through all the files to execute the regex functions
for i in range(len(num_of_chunks)):
    if i < 2:
        path = worker_path + str(i) + '.csv'
        output = worker_proc + str(i) + '.csv'
        worker(path, street_names_merge, strip_address, strip_st_type, regex_st_types, output)


# Opens all the processed files, and creaes a larger series
worker_proc = 'Data/worker_processed/'
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

print pad_mapper_clean
# Create an empty dataframe to hold the values to be queried from the Mapzen API
mapzen_df = pd.DataFrame(columns=['ADDR_NUM_mapozen', 'BASE_mapzen', 'COMMUNITY_mapzen', 'latitude', 'longitude', 'joint_addresses_mapzen','mapzen_confidence'])

