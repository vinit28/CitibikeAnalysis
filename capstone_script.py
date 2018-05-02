#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 13:13:54 2018

@author: vinit.shah1ibm.com
"""
###################### Load Libraries and Data ######################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import *
import seaborn as sns
from matplotlib import rcParams
import geopy.distance
import datetime as dt

df = read_csv('2017v2.csv')
df = df.drop(['Unnamed: 0'], axis = 1)
df.dtypes

df['Start Time'] = to_datetime(df['Start Time'])
df['Stop Time'] = to_datetime(df['Stop Time'])
df['Start Station Name'] = df['Start Station Name'].astype('category')
df['End Station Name'] = df['End Station Name'].astype('category')
df['User Type'] = df['User Type'].astype('category')
df['Gender'] = df['Gender'].astype('category')

df = df.drop(df.index[(df['Trip Duration'] > 2700)])
df = df[df['Distance'] < 30]
df = df[df['mile_hour'] < 20]
df = df[df['mile_hour'] > 0.1]
df_copy = df
df_sample = df.sample(frac = 0.1, random_state = 0)
'''
=================== Background, Context, and Objective ===================
Client: Mayor of NYC
Objective: Help the mayor get a better understanding of citibike ridership by 
           creating an operating report for 2017.

Ask: 
1)	Top 5 stations with the most starts (showing # of starts)
2)	Trip duration by user type
3)	Most popular trips based on start station and stop station)
4)	Rider performance by Gender and Age based on avg trip distance (station to station), 
    median speed (trip duration / distance traveled)
5)	What is the busiest bike in NYC in 2017? How many times was it used? How many minutes was it in use?
6)   A model that can predict how long a trip will take given a starting point and destination. 

First, let's minimize the work and load in the data set. Aggregate all the months and save the 
data as a csv to not require this to be done in the future.

The column names have spaces in them, would be great to remove them for working purposes. However, not necessary.
If I was working on a team or a long term project, I would ocnfigure column names a little bit differently
to make them easier to work with.
    
The dataset is massive, ~16mil rows. BigData tools would be helpful, however, most require you to pay
or have an enterprise license or a limited trial. Additionally, the data is very dirty. 
Different files have different column names, need to account for this.
Mayor de Blasio doesn't have a technical background. The graphs here are as simple yet informative 
as possible. I could've made more complicated plots, however, they would not be as informative for the mayor.
==========================================================================
'''

#Set working directory to correct folder, or add path
#Minimize work to load data. Can be done in other ways as well. There may be a more effecient way to do this. Haven't explored this too deeply. 
files = !ls *.csv #For Ipython only
df = concat([read_csv(f, header=None,low_memory=False) for f in files], keys=files) #header = None because some files don't have a header

#Let's see what we're working with
#df.head()
#df.dtypes
##Scrappy way to get column names in the header.
#df2 = pd.read_csv('jan.csv')
#df.columns = df2.columns
#del[df2]
#df = df[df[1] != 'starttime']#Could've just dropped duplicate rows here as well
#df = df[df[1] != 'Start Time']
#df.head()
#Saved csv to avoid running code above multiple times in the future
#df.to_csv('2017.csv')

###################### Code above only needs to be run once ######################
#load aggregated dataset to minimize time needed to run the code
df = read_csv('2017.csv')
df = df.drop(['Unnamed: 0', 'Unnamed: 1'], axis = 1)
list(df)

'''
========================= Part 1: Top 5 Stations =========================
Let's check if there's any noise or cleanup which needs to be done before creating the chart.
1. Any missing values?
    - Mostly for Birth year and a few for User Type. We can ignore these for now.
2. Let's get the data in the right format
    a. Trip Duration - Int
    b. Start Time - DateTime
    c. Stop Time - DateTime
    d. Start Station ID - Categorical
    e. Start Station Name - Categorical
    f. Start Station Latitude - 
    g. Start Station Longitude
    h. End Station ID - 
    i. End Station Name -
    j. End Station Latitude -
    k. End Station Longitude -
    l. Bike ID - 
    m. User Type - Categorical
    n. Birth Year - Ordinal
    o. Gender - Categorical
3. Any trips which lasted less than 1.5 minute (90 seconds). If so, in the ideal world, we 
   should not include this start, we may double count. If a bike is broken, a user will dock it
   again within a minute an pick-up another one.
   - Line 125-127 confirms this hypothesis! Would be ideal to not include any starts
   where a tip lasted less than 90 seconds *and* the start tation = end station. Line 129-130 handles this
4. Anomalies such as theft and broken docks shouldn't matter for this metric and can be dealth with later.
==========================================================================
'''
#Percentage of missing data. Will come in handy when modelling
def missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
missing_table = missing_data(df)

df.dtypes
#df['Trip Duration'] = df['Trip Duration'].astype(int)
df['Start Time'] = to_datetime(df['Start Time'])
df['Stop Time'] = to_datetime(df['Stop Time'])
df['Start Station ID'] = df['Start Station ID'].astype('category')
df['Start Station Name'] = df['Start Station Name'].astype('category')
#df['Start Station Latitude'] = 
#df['Start Station Longitude'] = 
df['End Station ID'] = df['End Station ID'].astype('category')
df['End Station Name'] = df['End Station Name'].astype('category')
#df['End Station Latitude'] = 
#df['End Station Longitude'] = 
#df['Bike ID'] = 
df['User Type'] = df['User Type'].astype('category')
#df['Birth Year'] = df['Birth Year'].astype(int)
df['Gender'] = df['Gender'].astype('category')

#Quasi Confirm Hypothesis in point #3
df_bikenum = pd.DataFrame()
df_bikenum['First Bike'] = df[df['Trip Duration'] < 90]['Start Station Name'] 
df_bikenum['Second Bike'] = df[df['Trip Duration'] < 90]['End Station Name']
df_bikenum.head()

df = df.drop(df.index[(df['Trip Duration'] < 90) & 
                          (df['Start Station Latitude'] == df['End Station Latitude'])])
top5 = pd.DataFrame() 
top5['Station']=df['Start Station Name'].value_counts().head().index
top5['Number of Starts']=df['Start Station Name'].value_counts().head().values

#Plot for Part 1
ax = sns.barplot('Station', 'Number of Starts', data = top5)
ax.set_title('Top 5 Citi Bike Stations by Number of Starts')
rcParams['figure.figsize'] = 12,10
ax.set_xticklabels(ax.get_xticklabels(),rotation=40, ha = 'right')
for index, row in top5.iterrows():
    ax.text(row.name,row['Number of Starts']+1500,row['Number of Starts'], color='black', ha="center", fontsize = 12)
plt.savefig('Top 5 Stations.png')

'''
=================== Part 2: Trip Duration by User Type ===================
This question is a bit unclear in terms of what to do with the anomalies, so 
I'll be making two graphs. One with anomalies, one without. 

1. There are NA values in the dataset for usertype as can be seen from missing_table.
   Since it's only 0.09% of the data, it's safe to remove.

According to Citi Bikes' website: The first 45 minutes of each ride is included 
                                  for Annual Members, and the first 30 minutes 
                                  of each ride is included for Day Pass users. 
                                  If you want to keep a bike out for longer, 
                                  it’s only an extra $4 for each additional 15 
                                  minutes.

It's safe to assume, no one (or very few people) will be willing to rent a bike 
for more than 2 hours. If they did, it would cost them an additional $20 assuming
they're annual subscribers. It would be more economical for them to buy a bike 
if they want that workout or use one of the tour bikes in central park if they
want to tour and explore the city on a bike. There may be a better way to choose 
an optimal cutoff, however, time is key in a client project. Or just docing and
getting another bike. The real cost of a bike is accrued ~24 hours. 

Anomalies: Any trip which lasts longer than 2 hours (7,200 seconds) probably indicates a stolen
bike or incorrect docking of the bike. As an avid Citibike user, I know first hand that it doesn't
make any sense for one to use a bike for more than one hour! However, I've added a one hour cushion
just in case. No rider woul dplan to go over the maximum 45 minutes allowed. In case they choose 
destinations which would take more than an hour to bike (brooklyn to washington heights), I've added
a cushion. However, I would be comfortable reducing this to one hour in the future.
1st graph - with anomalies in dataset
a. The graph under ax2 is a bargraph of average trip duration for each user type.
    It's helpful, but would be better to see a boxplot and get an idea of the distribution.
b. Line 196 is a basic Boxplot based with anomalies included. As we can see, there's
    too much noise for this to be useful. It'll be better to look at this without
    anomalies.
2nd graph - without anomalies in dataset
a. Still not useful, let's add a column with minutes for trip Duration.
b. Boxplot with minutes is much more useful. There are still some outliers, however, 
    it is informative 
==========================================================================
'''
df = df.dropna(subset=['User Type']) 
df['User Type'].value_counts()

TD_user = pd.DataFrame()
TD_user['Avg. Trip Duration'] = df.groupby('User Type')['Trip Duration'].mean()
TD_user['User Type'] = TD_user.index

#Average trip Duration per User Type
ax2 = sns.barplot('User Type', 'Avg. Trip Duration', data = TD_user)
ax2.set_title('Average Trip Duration by User Type')
rcParams['figure.figsize'] = 12,10
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=40, ha = 'right')
for index, row in TD_user.iterrows():
    ax2.text(row.name,row['Avg. Trip Duration']+50,(str(round(row['Avg. Trip Duration'],2))+"  Seconds"), 
             color='black', ha="center", fontsize = 10)
plt.savefig('Avg Trip by User.png')

#Boxplots are more informative to visualize breakdown of data
df.boxplot('Trip Duration', by = 'User Type')

#Remove anomalies based on definition above
df = df.drop(df.index[(df['Trip Duration'] > 7200)])

#Boxplots are more informative to visualize breakdown of data
df.boxplot('Trip Duration', by = 'User Type')

#Add Minutes column for Trip Duration
df['Minutes'] = df['Trip Duration']/60
#For Visual purposes, rounded
df['Minutes'] = round(df['Minutes'])
df['Minutes'] = df['Minutes'].astype(int)
df['Minutes'].head()

#Final Boxplot with some outliers. Could turn of outliers with showfliers = False
df.boxplot('Minutes', by = 'User Type')
df.boxplot('Minutes', by = 'User Type', showfliers = False)

TD_user2 = pd.DataFrame()
TD_user2['Avg. Trip Duration'] = df.groupby('User Type')['Minutes'].mean()
TD_user2['User Type'] = TD_user.index
#TD_user2.to_csv('minutes.csv')
#Average Trip Duration Based on Minutes
ax3 = sns.barplot('User Type', 'Avg. Trip Duration', data = TD_user2)
ax3.set_title('Average Trip Duration by User Type')
#rcParams['figure.figsize'] = 12,10
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=40, ha = 'right')
for index, row in TD_user2.iterrows():
    ax3.text(row.name,row['Avg. Trip Duration']+0.5,(str(round(row['Avg. Trip Duration'],2))+"  Minutes"), 
             color='black', ha="center", fontsize = 10)
plt.savefig('Avg Trip by User_2.png')

#Undo rounding for modelling purposes
df['Minutes'] = df['Trip Duration']/60

'''
======================== Part 3: Most Popular Trip ========================
To get most popular trips, the most convenient way to do this is by using the group
function in pandas. It's analogous to a Pivot table.

The groupby function makes it extremely easy and convenient to identify the most
popular trips. Visuals and transformations can be found below.

===========================================================================
'''
trips_df = pd.DataFrame()
trips_df = df.groupby(['Start Station Name','End Station Name']).size().reset_index(name = 'Number of Trips')
trips_df = trips_df.sort_values('Number of Trips', ascending = False)
trips_df["Start Station Name"] = trips_df["Start Station Name"].astype(str)
trips_df["End Station Name"] = trips_df["End Station Name"].astype(str)
trips_df["Trip"] = trips_df["Start Station Name"] + " to " + trips_df["End Station Name"]

trips_df.to_csv('Most Popular Trips.csv')

trips_df = trips_df[:10]
trips_df = trips_df.drop(['Start Station Name', "End Station Name"], axis = 1)

ax4 = sns.barplot('Trip','Number of Trips', data = trips_df)
ax4.set_xlabel("Trip",fontsize=25)
ax4.set_ylabel("Number of Trips",fontsize=25)
ax4.set_title('Most Popular Trips',size = 35)
ax4.tick_params(labelsize=25)
ax4.set_xticklabels(ax4.get_xticklabels(),rotation=40, ha = 'right', fontsize = 20)
rcParams['figure.figsize'] = 50,25
for index, row in trips_df.iterrows():
    ax4.text(row.name,row['Number of Trips']-180,row['Number of Trips'], 
             color='black', ha="center", fontsize = 25)
plt.savefig('Most Popular Trips.png',bbox_inches='tight')


'''
================ Part 4: Rider Performance by Gender and Age ================
Ask: Rider performance by Gender and Age based on avg trip distance (station to station), 
median speed (trip duration / distance traveled)

Let's make sure the data we're working with here is clean.

1. Missing Gender and Birth Year values - Check missing_table
    - No for Gender. Yes for Birth Year
    - ~10% Missing Birth year. Not a big chunk of data. Can either impute missing 
    values or drop it. Since it's less than 10% of the data, it's safe to assume the rest
    of the 90% is a representative sample of data and we can replace the birth year with 
    the median, based on gender and Start Station ID. This will be done after anomalies are 
    removed and speed is calculated.
2. Are there anomalies?
    - For Birth Year, there are some people born prior to 1956. I can believe some 
    60 year olds can ride a bike and that's a stretch, however, anyone "born"
    prior to that riding a citibike is an anomaly and false data. There could be a few 
    senior citizens riding a bike, but probably not likely.
    - My approach is to identify the age 2 standard deviations lower than the mean.
    After calculating this number, mean-2stdev, I removed the tail end of the data, birth year prior to 1956.
3. Caulculate an Age column to make visuals easier to interpret.
4. Calculate trip distance (Miles)
    - No reliable way to calculate bike route since we can't know what route a 
    rider took without GPS data from each bike. 
    - Could use Google maps and use lat,long coordinates to find bike route distance.
    However, this would require more than the daily limit on API calls. Use the 
    geopy.distance packge which uses Vincenty distance uses more accurate 
    ellipsoidal models. this is more accurate than Haversine formula, but doesn't matter
    much for our purposes.
5. Caulculate Speed (min/mile) and (mile/hr)
    - (min/mile) - Can be used like sprint time (how fast does this person run)
    - (mile/hr) - Conventional approach, but units may be difficult to understand)
    - Miles/hour is an easy to understand unit of measure and one most people are used
    to seeing. So the visual will be created based on this understanding.
6. Dealing with "circular" trips
    - Circular trips are trips which start and end at the same station. The distance
    for these trips will come out to 0, however, that is not the case. These points
    will skew the data and visuals. Will be removing them to account for this issue.
    - For the model, this data is also irrelevant. Because if someone is going on a 
    circular tri, the only person who kknows how long the trip is going to take is the
    rider themself, assuming they know that. So it's safe to drop this data for the model.
7. Rename Gender Values in Legend from 0,1,2 to Unknown, Male, Female, respectively.
    - The rows where Gender is unknown throws the visual off. There are two ways to handle this
        - Remove the missing data. This would not result in a significant loss of information since only 58073 rows 
        have gender as unknown.
        - We can impute missing values, however given the proportion of unknowns the information gain would be
        negligible.
        - Based on the reasons above, I've decided to remove data with unknown gender. These rows should not have
        a significant imact on the predictive model later on. However, I will confirm this.
8. Determine Gender and Age performance based on Average Trip distance
    - Similar to graphs for speed. Pretty straightforward.
 
===========================================================================
'''

df['Gender'].value_counts()
#df['Birth Year'].mean()-(2*df['Birth Year'].std())
df = df.drop(df.index[(df['Birth Year'] < 1956)])

df['Start Coordinates'] = list(zip(df['Start Station Latitude'], df['Start Station Longitude']))
df['End Coordinates'] = list(zip(df['End Station Latitude'], df['End Station Longitude']))

#In the future, for a dataset of this size, I would consider using the Haversine formula to calculate distance if it's faster.
dist = []
for i in range(len(df)):
    dist.append(geopy.distance.vincenty(df.iloc[i]['Start Coordinates'],df.iloc[i]['End Coordinates']).miles)
    if (i%1000000==0):
        print(i)

df['Distance'] = dist
df['Distance'].head()
#df['Distance'].to_csv('Distance.csv')
df.dtypes
df.head()

df['min_mile'] = round(df['Minutes']/df['Distance'], 2)
df['mile_hour'] = round(df['Distance']/(df['Minutes']/60),2)
df.describe()
#Replace missing birth year by median based on speed and gender
df['Birth Year'] = df.groupby(['Gender','Start Station ID'])['Birth Year'].transform(lambda x: x.fillna(x.median()))

df['Birth Year'].isnull().sum()
#Still have a few nulls, but it's only 2342 entries now. Comfortable dropping these.
df = df.dropna(subset=['Birth Year'])

df['Age'] = 2018 - df['Birth Year']
df['Age'] = df['Age'].astype(int)

df = df.drop(df.index[(df['Distance'] == 0)])

#Saving updated DataFrame in case Kernel crashes during distance calculation
df.to_csv('2017v2.csv')
#=======================================================Data Save==========================================================================
#=======================================================Data Save=================================================================================================================================
#=======================================================Data Save=================================================================================================================================
df = read_csv('2017v2.csv')
df = df.drop(['Unnamed: 0'], axis = 1)
df.dtypes
#The following loop takes forever to run. If you're running this code, please just read in the Distance.csv file
dist = pd.read_csv('Distance.csv', header = None)
dist = dist.drop([0], axis = 1)


df['Distance'] = df.groupby(['Gender','Start Station ID'])['Distance'].transform(lambda x: x.fillna(x.median()))
df_copy = df
#temporarily 0, not imputing missing values because we can't use it in the model anyway
df['mile_hour']=df['mile_hour'].fillna(df['mile_hour'].median())
df[df['mile_hour']>40]
df[df['mile_hour']<40].count()
df.iloc[1995]
missing_data(df)

df = df[df['Distance'] < 30]
df = df[df['mile_hour']>40]

#Dropping unknown to make the visual more informative. Unknown gender may be important for the model, which is why I created a copy of the original dataframe.
df1 = df.drop(df.index[(df['Gender'] == 0)])

#Min/Mile
fig, ax5 = plt.subplots(figsize=(11,5))
df1.groupby(['Age','Gender']).median()['min_mile'].unstack().plot(ax=ax5)
ax5.legend(['Male','Female'])
plt.ylabel('Median Speed (min/mile)')
plt.title('Rider Performance Based on Gender and Age (Median Speed in min/mile)')

#Miles/hr
fig1, ax6 = plt.subplots(figsize=(11,5))
df_sample.groupby(['Age','Gender']).median()['mile_hour'].unstack().plot(ax=ax6)
#ax6.legend(['Female', 'Male'])
plt.ylabel('Median Speed (miles/hr)')
plt.title('Rider Performance Based on Gender and Age (Median Speed in miles/hr)')

#Averge Distance
fig2, ax7 = plt.subplots(figsize=(11,5))
df1.groupby(['Age','Gender']).mean()['Distance'].unstack().plot(ax=ax7)
ax7.legend(['Male', 'Female'])
plt.ylabel('Average Distance (miles)')
plt.title('Rider Performance Based on Gender and Age (Average Distance in Miles)')

'''
================ Part 5: Busiest Bike by Times and Minutes Used ================
Ask: 1. What is the busiest bike in NYC in 2017? 
        - Bike 25738
     2. How many times was it used?
        - 2049 times
     3. How many minutes was it in use?
        - 25,754 Minutes

1. Busiest bike and count can be identified by a groupby function
2. Function above will also identify the number of times the bike was used
3. A similar groupby function which calls for the sum on minutes can identify
    the number of minutes the bike was used. 

================================================================================
'''
#Bike usage based on number of times used
bike_use_df = pd.DataFrame()
bike_use_df = df.groupby(['Bike ID']).size().reset_index(name = 'Number of Times Used')
bike_use_df = bike_use_df.sort_values('Number of Times Used', ascending = False)
#bike_use_df.to_csv('Q5.csv')
bike_use_df = bike_use_df[:10]
bike_use_df['Bike ID'] = bike_use_df['Bike ID'].astype(str)
bike_use_df['Bike ID'] = ('Bike ' + bike_use_df['Bike ID'])

ax8 = sns.barplot('Number of Times Used', 'Bike ID',data = bike_use_df)
ax8.set_title('Most Popular Bikes by Number of Times Used')
rcParams['figure.figsize'] = 12,10
for index, row in bike_use_df.iterrows():
    ax8.text(row['Number of Times Used']-90,index,row['Number of Times Used'], 
             color='black', ha="center")
plt.savefig('Most used bike.png',bbox_inches='tight')

#Bike usage based on minutes used
bike_min_df = pd.DataFrame()
bike_min_df['Minutes Used'] = df.groupby('Bike ID')['Minutes'].sum()
bike_min_df = bike_min_df.reset_index()
bike_min_df = bike_min_df.sort_values('Minutes Used', ascending = False)
bike_min_df['Bike ID'] = bike_min_df['Bike ID'].astype(str)
bike_min_df['Bike ID'] = ('Bike ' + bike_min_df['Bike ID'])

bike_min_df = bike_min_df[:10]
#bike_min_df.to_csv('Min_Bike.csv')
ax9 = sns.barplot('Minutes Used', 'Bike ID',data = bike_min_df)
ax9.set_title('Most Popular Bikes by Minutes Used')
for index, row in bike_min_df.iterrows():
    ax9.text(row['Minutes Used']-1500,index,row['Minutes Used'], 
             color='black', ha="center")
plt.show()
plt.savefig('Most used bike.png',bbox_inches='tight')



'''
========================== Part 6.1: Predictive Model - Baseline Model ===========================
Ask: Build a model that can predict how long a trip will take given a starting 
     point and destination. 

***Assumptions on how the Kiosk will work: After speaking to Daniel Yawitz (if you're 
looking at this, thanks for the clarification), I was told that we should assume that when
a user inputs the start and end station, they swipe their key fob (if they're a subscriber)
and enter their info on the kiosk (if they're a "Customer") prior to entering the start 
and end station. This means that we would know their gender and age. Thus these variables
can be used in building the model.***

Step 1: - This dataset is massive. Almost 14 million rows. Let's work on a *random*
          subsample while we build and evaluate models. If I tried to build and 
          evaluate a model on the entire dataset, each run would take me ~10+ minutes
          depending on the model. 
          One good way to decide what portion of your data to work with is using
          a learning curve. However, my kernel keeps crashing while trying to 
          create that learning curve. However, given the size of the data and from 
          experience by working with senior data scientists on projects with BAML 
          and other firms I know that I can comfortably work with a few thoushand 
          rows of data given the fact that this is only one year of data. If we were
          working with data for multiple years, I'd need to reconsider this approach.
          However, given the reasons above, I've decided to sample 10% of the data.
          It's stil ~1.3 million rows and should be a representative sample since
          it's randomly selected. I'll be evaluating my model on df_sample. 
        - I also made the same visuals on the sample as I did on 
          
          
Step 2: - Let's get a baseline. If I were to just run a simple multi-variate 
          linear regression, what would my model look like and how accurate would 
          it be? Need to prepare the data for a multivariate regression
            1) Drop irrelevant columns
                - Trip Duration: We have the minutes column, which is the target variable
                - Stop Time: In the real world, we won't have this information when 
                             predicting the trip duration.
                - Start Station ID: Start Station Name captures this information
                - Start Station Latitude: Start Station Name captures this information
                - Start Station Longitude: Start Station Name captures this information
                - Start Coordinates: Start Station Name captures this information
                - End Station ID: End Station Name captures this information
                - End Station Latitude: End Station Name captures this information
                - End Station Longitude: End Station Name captures this information
                - End Coordinates: End Station Name captures this information
                - Bike Id: We won't know what bike the user is going to end up using
                - Min_Mile: Effectively the same information as end time when combined with distance. 
                            We won't have this information in the real world.
                - mile_hour: Effectively the same information as end time when combined with distance. 
                            We won't have this information in the real world.
                    (Speed * Distance = Trip Duration): Which is why speed is dropped
                - Birth Year: Age captures this information
                - Start Station Name and End Station Name: The distance variable 
                  captures the same information. For the model, if a user is inputting
                  start and end station, we can build a simple function to calculate the
                  distance which would capture the same information.
            2) Basic cleaning of data FOR NOW.This is only being done for the baseline model
                - Start Time: Requires reformatting. Will do this after baseline model
                - Dumify categorical variables
                - Scale Age
                - Don't scale distance, since it does not just represent distance, but
                  is also indicative of the trip the rider is making (start and station)

=================================================================================================
'''
#Residual Plot
sns.residplot(df_basemodel['Distance'], df_basemodel['Minutes'])
#This plot points out some clear outliers in the data. Upon further analysis, we can see that some distances are more than 5000 miles. 
#Let's delete this data since it's clearly an outlier. Start coordinates for these locations is (0.0,0.0). Something is off here.
#30 miles, because the two furthest stations in NYC (Jersey to Queens are less far than 30 miles). Could've used a more realistic cutoff.
missing_data(df)
df['Birth Year'] = df.groupby(['Gender','Start Station ID'])['Birth Year'].transform(lambda x: x.fillna(x.median()))
sns.residplot(df_basemodel['Age'], df_basemodel['Minutes'])

#Random Sample of Data
df_sample = df.sample(frac = 0.1, random_state = 0)
df.dtypes

round(df.describe(),2)

#BUILD A BASELINE MODEL

#Drop Irrelevant data
def drop_data(df):
    df = df.drop(['Trip Duration','Stop Time','Start Station ID','Start Station Latitude','Start Station Longitude',
                  'Start Coordinates','End Station ID', 'End Station Latitude', 'End Station Longitude', 
                  'End Coordinates','Bike ID', 'Start Station Name','Birth Year','End Station Name','min_mile', 
                  'mile_hour', 'Age'], axis = 1)
    return df

df_basemodel = drop_data(df_sample)
df_basemodel = df_basemodel.drop('Start Time', axis =1)
df_basemodel.dtypes
df_basemodel.corr().loc[:,'Minutes']
#Dummify categorical data and avoid 
df_basemodel = pd.get_dummies(df_basemodel, drop_first = True)

'''
#Scale Age
from sklearn.preprocessing import StandardScaler
sc_bm = StandardScaler()
df_basemodel['Age'] = sc_bm.fit_transform(df_basemodel[['Age']])

#Get the target variable at the end. Will be handy later on.
def reorder_columns(df):
    cols = df_basemodel.columns.tolist()
    cols = cols[:2]+cols[3:]
    cols.append('Minutes')
    df=df[cols]
    return df

df_basemodel = reorder_columns(df_basemodel)
'''
df_basemodel.dtypes

#Train Test Split
#Predictor variable
X = df_basemodel.iloc[:,1:]
#Target variable
y = df_basemodel.iloc[:,0]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Fit Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.score(X_test,y_test)

#Using Statsmodel because it has the summary function.
import statsmodels.api as sm
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
regressor_OLS = sm.OLS(y_train, X_train).fit()
regressor_OLS.summary()

#Prediction for test set
y_pred = regressor_OLS.predict(X_test)

'''
# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test.head(10), y_pred[:10], color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

#Regression Plots
sns.regplot(x = "Age", y = "Minutes", data = df_basemodel)
plt.ylim(0,)
'''

'''
========================== Part 6.2: Predictive Model - Including Date ===========================
Baseline Model: Adjusted R^2 and R^2 = 55.9% (depending on random_state)        
1) The baseline model is ok, but nothing spectacular. The R_Squared and Adjusted R_Squared is 
    pretty much the same, 53.6%. The F-Stat is also 0.00 which is a good sign. The gender female's 
    p-value is high, however, we can't loose this information due to the fact that this is a categorical
    variable and we'd be loosing substantial information by dropping it. 
2) Steps to make improvements:
    a) Add back time in the following format
        - Is the ride on a WEEKDAY or WEEKEND. Weekday, is rush-hour commute for the most part and
          probably from home to work. Weekend could be a longer, more casual ride and have higher variability.
        - Is the ride in the MORNING, AFTERNOON, EVENING, or NIGHT. The exact timing will be based on the 
          difference in trip duration based on time of day. Will have visuals below.
        - What season is it?
            - December - Feb. = Winter
            - March - May = Spring
            - June - Aug. = Summer
            - Sept. - Nov. = Fall
3) Evaluate model and check CORRELATION to ensure against collinearity and identify what's going on

=================================================================================================
'''
#The following code was used to create get_date_info
'''
#Weekday or Weekend: 1 = Weekday, 0 = Weekend
df_model1['d_week'] = (df_model1['d_week']<5).astype(int)

#Season of the year: Winter = 0, Spring = 1, Summer = 2, Fall = 3
df_model1['m_yr'] = df_model1['m_yr'].replace(to_replace=[12,1,2], value = 0)
df_model1['m_yr'] = df_model1['m_yr'].replace(to_replace=[3,4,5], value = 1)
df_model1['m_yr'] = df_model1['m_yr'].replace(to_replace=[6,7,8], value = 2)
df_model1['m_yr'] = df_model1['m_yr'].replace(to_replace=[9,10,11], value = 3)

#Visualize difference in ridership based on time of day
#Ploted both mean and media, similar pattern, using median due to the high variance in the data
fig3, ax10 = plt.subplots(figsize=(15,7))
df_model1.groupby('ToD').mean()['Minutes'].plot(ax=ax10)
#df_model1.groupby('ToD').median()['Minutes'].plot(ax=ax10)
plt.ylabel('Median Minutes per Trip')
plt.xlabel('Hour of the Day')
plt.title('Median Minutes per Trip by Time of Day')
plt.show()

#Based on the visual above and distribution of ridership: NIGHT = 20-5, MORNING = 5-9, AFTERNON = 9-14, EVENING = 14-20
df_model1['ToD'] = pd.cut(df_model1['ToD'], bins=[-1, 5, 9, 14, 20, 25], labels=['Night','Morning','Afternoon','Evening','Night1'])
df_model1['ToD'] = df_model1['ToD'].replace(to_replace='Night1', value = 'Night')
df_model1['ToD'] = df_model1['ToD'].cat.remove_unused_categories()

#Convert m_yr to categorical
df_model1['m_yr'] = df_model1['m_yr'].astype('category')


#Dummify categorical data and avoid 
df_model1 = pd.get_dummies(df_model1, drop_first = True)
'''

def get_date_info(df):
    df['d_week'] = df['Start Time'].dt.dayofweek
    df['m_yr'] = df['Start Time'].dt.month
    df['ToD'] = df['Start Time'].dt.hour

    df['d_week'] = (df['d_week']<5).astype(int)

    df['m_yr'] = df['m_yr'].replace(to_replace=[12,1,2], value = 0)
    df['m_yr'] = df['m_yr'].replace(to_replace=[3,4,5], value = 1)
    df['m_yr'] = df['m_yr'].replace(to_replace=[6,7,8], value = 2)
    df['m_yr'] = df['m_yr'].replace(to_replace=[9,10,11], value = 3)
    
    df['ToD'] = pd.cut(df['ToD'], bins=[-1, 5, 9, 14, 20, 25], labels=['Night','Morning','Afternoon','Evening','Night1'])
    df['ToD'] = df['ToD'].replace(to_replace='Night1', value = 'Night')
    df['ToD'] = df['ToD'].cat.remove_unused_categories()
    
    df['m_yr'] = df['m_yr'].astype('category')
    df['d_week'] = df['d_week'].astype('category')

    return(df)

df_model1 = drop_data(df_sample)
df_model1 = get_date_info(df_model1)
df_model1 = df_model1.drop('Start Time', axis =1)

df_model1.dtypes
df_model1 = pd.get_dummies(df_model1, drop_first = True)
df_model1.dtypes


#Let's get the model cranking
#Scale Age
#from sklearn.preprocessing import StandardScaler
sc_m1 = StandardScaler()
df_model1['Age'] = sc_m1.fit_transform(df_model1[['Age']])


#Train Test Split
#Predictor variable
X = df_model1.iloc[:,1:]
#Target variable
y = df_model1.iloc[:,0]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fit Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.score(X_test,y_test)

#Using Statsmodel because it has the summary function.
import statsmodels.api as sm
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
regressor_OLS = sm.OLS(y_train, X_train).fit()
regressor_OLS.summary()

df_model1.corr().iloc[:,0]
'''
========================== Part 6.3: Predictive Model - Improving Model 1 ===========================
Model 1: Negligible improvement in R^2: 67.5% (depending on random_state used)

    - Date seems to not have as big of an impact. May be more useful to split time and day differently, but
    acording to the graphs, there is no indication of a better split. Next steps will be to factor in speed
    and distance based on Gender and Trip. By not being able to encode start and end stations (due to the
    sheer number of points), we are losing crucial information on the trip itself. We need another proxy for
    those measures. 
    - Another change I could make is to bin age into buckets. However, the data indicates that age has no correlation
    or effect on the trip duration. This is counter-intuitive, however, I don't have a good reason to refute the data.

Next Steps:
    1. Include Average Speed based on: Trip and User Type
        - Reason for Trip: Some trips are up hill, others are down hill. Some routes, such as through times 
                           square involved heavy trafic, based on intuition.
        - Reason for User Type: Tourists (Customers), will usually ride more slowly with frequent stops than 
                                a Subscriber, according to the data.
    2. Include average duration for each trip based on: Trip and User Type
        - Reason, for each, same as above.
        - Using seconds instead of minutes for granularity in information.

    3. A bit of inexperience here:
        - The imputed columns are "dependent" variables in the since speed is derived from trip duration and distance. 
        There may be a better way to do this, however, I would need to consult someone. Would appreciate feedback on this.
        - I don't think it should be a big issue that they're "dependent" since they serve more as anchor points. 
        But I'd appreciate feeback.

=====================================================================================================
'''

#Code below used to create get_speed_distance
'''
df_model2 = df_sample

df_model2['Start Station Name'] = df_model2['Start Station Name'].astype(str)
df_model2['End Station Name'] = df_model2['End Station Name'].astype(str)
df_model2['Trip'] = df_model2['Start Station Name'] + ' to ' + df_model2['End Station Name']
df_model2['Trip'] = df_model2['Trip'].astype('category')

df_model2['avg_speed'] = df_model2.groupby(['Trip','User Type'])['mile_hour'].transform('mean')
df_model2['avg_duration'] = df_model2.groupby(['Trip','User Type'])['Trip Duration'].transform('median')
#Code to make sure transfrom function works appropriately.
#df_model2[df_model2['avg_speed']==df_model2.loc[258881]['avg_speed']]
'''
df = df.drop(df.index[(df['Trip Duration'] > 2700)])

def get_speed_distance(df):

    df['Start Station Name'] = df['Start Station Name'].astype(str)
    df['End Station Name'] = df['End Station Name'].astype(str)
    df['Trip'] = df['Start Station Name'] + ' to ' + df['End Station Name']
    df['Trip'] = df['Trip'].astype('category')
    
#    df['avg_speed'] = df.groupby(['Trip','User Type'])['mile_hour'].transform('mean')
    df['avg_duration'] = df.groupby(['Trip','User Type'])['Trip Duration'].transform('median')
    
    return df

#Drop Irrelevant data, updated with Trip
def drop_data_trip(df):
    df = df.drop(['Trip Duration','Stop Time','Start Station ID','Start Station Latitude','Start Station Longitude',
                  'Start Coordinates','End Station ID', 'End Station Latitude', 'End Station Longitude', 
                  'End Coordinates','Bike ID', 'Start Station Name','Birth Year','End Station Name','min_mile', 
                  'mile_hour', 'Age', 'Trip'], axis = 1)
    return df

df_model2 = get_speed_distance(df_sample)
df_model2 = drop_data(df_model2)
df_model2 = get_date_info(df_model2)
df_model2 = df_model2.drop('Start Time', axis =1)

df_model2.dtypes
df_model2 = pd.get_dummies(df_model2, drop_first = True)
df_model2.dtypes

#Train Test Split
#Predictor variable
X = df_model2.iloc[:,1:]
#Target variable
y = df_model2.iloc[:,0]
#Split
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fit Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.score(X_test,y_test)

#Using Statsmodel because it has the summary function.
import statsmodels.api as sm
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
regressor_OLS = sm.OLS(y_train, X_train).fit()
regressor_OLS.summary()

df_model2.corr().loc[:,'Minutes']

#Ensure model accuracy with cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(regressor, X_train, y_train, cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

'''
========================== Part 6.4: Predictive Model - Improving Model 2 ===========================
The model above is extremely good. But can I make it even better?

- One of the grading requirements for the project is to include an additional data source. Weather and traffic
make the most sense. I could include pedestrian traffic data from citibike's website, however, I don't think it'll
be too helpful due to the fact that there are now bikelanes in NYC. I could use weather data as well.
    - Concerns with using weather data:
        - Weather dictates wether or not a rider will bike, not how long they will bike. For weather to determine
        how long one will bike, it would have to dictate where they go. However, since grading deends on my ability
        to use external data, I'm going to test this hypothesis. If weather is not a strong indicatior, I will remove
        it in the next model. 

=====================================================================================================
'''
df_weather = pd.read_csv('weather2017.csv')
df_weather = df_weather.iloc[:,1:5]
df_weather['DATE'] = to_datetime(df_weather['DATE'])

df_model3 = get_speed_distance(df_sample)
df_model3 = drop_data(df_model3)

#Following code was used to create get_weather
'''
df_model3['DATE'] = to_datetime(df_model3['Start Time'].dt.date)


df_w = df_weather
df_mod = df_model3    

df_w.head()
df_mod.dtypes
test = df_mod.merge(df_w, on = 'DATE', how = 'left')
test.head()

df = df_model3.merge(df_weather, on = 'DATE', how = 'left')
'''

def get_weather(df):
    df['DATE'] = to_datetime(df['Start Time'].dt.date)
    
    df = df.merge(df_weather, on = 'DATE', how = 'left')
    return df

df_model3 = get_date_info(df_model3)
df_model3 = get_weather(df_model3)
df_model3 = df_model3.drop(['Start Time','Trip','DATE'], axis =1)

df_model3.dtypes
df_model3 = pd.get_dummies(df_model3, drop_first = True)
df_model3.dtypes

from sklearn.preprocessing import StandardScaler
sc_m1 = StandardScaler()
df_model3['avg_duration'] = sc_m1.fit_transform(df_model3[['avg_duration']])


#Train Test Split
#Predictor variable
X = df_model3.iloc[:,1:]
#Target variable
y = df_model3.iloc[:,0]
#Split
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fit Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.score(X_test,y_test)

#Using Statsmodel because it has the summary function.
import statsmodels.api as sm
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
regressor_OLS = sm.OLS(y_train, X_train).fit()
regressor_OLS.summary()


df_model3.corr().loc[:,'Minutes']


df_model3 = get_speed_distance(df_sample)
df_model3 = drop_data(df_model3)
df_model3 = get_date_info(df_model3)
df_model3 = get_weather(df_model3)
df_model3 = df_model3.drop(['Start Time','Trip','DATE','ToD','m_yr','PRCP','TMAX','TMIN'], axis =1)

df_model3.dtypes
df_model3 = pd.get_dummies(df_model3, drop_first = True)


from sklearn.preprocessing import StandardScaler
sc_m1 = StandardScaler()
df_model3['avg_duration'] = sc_m1.fit_transform(df_model3[['avg_duration']])

df_model3.dtypes

#Train Test Split
#Predictor variable
X = df_model3.iloc[:,1:]
#Target variable
y = df_model3.iloc[:,0]
#Split
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fit Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.score(X_test,y_test)

#Using Statsmodel because it has the summary function.
import statsmodels.api as sm
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
regressor_OLS = sm.OLS(y_train, X_train).fit()
regressor_OLS.summary()

from sklearn.ensemble import RandomForestRegressor

regressor_rf = RandomForestRegressor(n_estimators = 80, random_state = 0, min_samples_leaf = 600)
regressor_rf.fit(X_train,y_train)
regressor_rf.score(X_test,y_test)

'''
========================== Part 6.5: Predictive Model - Final Model  ===========================
- As I thought, weather is not a major contributor to the model. Thus, I'll be taking out weather. Additionally,
I'll be taking out date attributes as mentioned earlier. 
    1. The ensemble algorithm, Random Forest takes much longer to run (even with a low n_estimators). If the run
    time is 5 minutes on ~1.5 million rows, I don't want to take the risk on ~15 million rows of data. 
    2. I could've used XGboost and other fancy algorithms, however, for a dataset of this size, it would take too
    long to run and the gains wouldn't be worth it if there are any. 
    3. Final Model:
        - Linear Regression:
            - Predictors: Distance, Gender, Average Duration based on Trip and Gender, User Type
================================================================================================
'''

def prep_data(df):
    df = get_speed_distance(df)
    df = drop_data(df)
    df = df.drop(['Start Time','Trip'], axis =1)
    return df

df_final = prep_data(df)

df_final.dtypes

df_final = pd.get_dummies(df_final, drop_first = True)


from sklearn.preprocessing import StandardScaler
sc_m1 = StandardScaler()
df_final['avg_duration'] = sc_m1.fit_transform(df_final[['avg_duration']])



#Train Test Split
#Predictor variable
X = df_final.iloc[:,1:]
#Target variable
y = df_final.iloc[:,0]
#Split
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fit Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.score(X_test,y_test)

#Using Statsmodel because it has the summary function.
import statsmodels.api as sm
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
regressor_OLS = sm.OLS(y_train, X_train).fit()
regressor_OLS.summary()


#Ensure model accuracy with cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(regressor, X_train, y_train, cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))