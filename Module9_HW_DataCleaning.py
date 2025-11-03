import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

flights_data = pd.read_csv('flights.csv') #This function reads the csv file and creates a pandas dataframe for flight data.
flights_data.head(10) #This function displays the first 10 rows of the dataframe as a preview. Otherwise 5 rows are shown by default.
flights_data_np = flights_data.to_numpy() #This function converts the pandas dataframe to a numpy array for flight data. (no column names)
weather_data = pd.read_csv('weather.csv') #This function reads the csv file and creates a pandas dataframe for weather data.
weather_data_np = weather_data.to_numpy() #This function converts the pandas dataframe to a numpy array for weather data. (no column names)

def check_missing_data(df, column_name):
    '''
    This function checks the percentage of missing data in a specified column of a dataframe.
    This is useful for data cleaning and analysis.
    Inputs:
        df: pandas dataframe
        column_name: string, name of the column to check for missing data
    Output:
        missing_percentage: float, percentage of missing data in the specified column rounded to 2 decimal places
    '''
    missing_count = df[column_name].isnull().sum()
    total_count = len(df[column_name])
    missing_percentage = round((missing_count / total_count) * 100, 2)
    return missing_percentage

#Question 1 How many flights were there from JFK to SLC? Int
flights_JFK_SLC = flights_data[(flights_data['origin'] == 'JFK') & (flights_data['dest'] == 'SLC')]
num_flights = len(flights_JFK_SLC)
print(f"\nQuestion 1: There were {num_flights} flights from JFK to SLC.")

#Question 2 How many airlines fly to SLC? Should be int
flights_to_SLC = flights_data[flights_data['dest'] == 'SLC']
airlines_to_SLC = flights_to_SLC['carrier'].nunique()
print(f"\nQuestion 2: There are {airlines_to_SLC} airlines flying to SLC.")

#Question 3 What is the average arrival delay for flights to RDU? float
flights_to_RDU = flights_data[flights_data['dest'] == 'RDU']
arrival_delays_RDU = flights_to_RDU['arr_delay']
missing_data_percent = check_missing_data(flights_to_RDU, 'arr_delay') #Checking for missing arrival delay data
print(f"\n\tQ3 data quality check - % missing data: {missing_data_percent}%") #This is just a data quality check!
# Since the missing data is minimal (4.8%), I am going to assume the delay is 0 and calculate the average accordingly.
average_arrival_delay = arrival_delays_RDU.fillna(0).mean()
print(f"\nQuestion 3: Average arrival delay for flights to RDU is {round(average_arrival_delay, 2)} minutes.")

#Question 4 What proportion of flights to SEA come from the two NYC airports (LGA and JFK)?  float
flights_to_SEA = flights_data[flights_data['dest'] == 'SEA']
NYC_flights_to_SEA = flights_to_SEA[(flights_to_SEA['origin'] == 'LGA') | (flights_to_SEA['origin'] == 'JFK')]
proportion_NYC_to_SEA = round(100 * len(NYC_flights_to_SEA) / len(flights_to_SEA), 2)
print(f"\nQuestion 4: {proportion_NYC_to_SEA}% of flights to SEA come from NYC airports.")

#Question 5 Which date has the largest average depature delay? Pd slice with date and float
#please make date a column. Preferred format is 2013/1/1 (y/m/d)
flights_data['date'] = pd.to_datetime(flights_data[['year', 'month', 'day']], format='%Y/%m/%d') #Creating a new datetime column called 'date' in the dataframe
missing_dep_delay_data = check_missing_data(flights_data, 'dep_delay') #Checking for missing departure delay data
print(f"\n\tQ5 data quality check - % missing departure delay data: {missing_dep_delay_data}%") #This is just a data quality check!
# Since the missing departure delay data is very small (2.45%), I will assume the delay is 0 for those entries.
data_of_interest = flights_data[['date', 'dep_delay']].fillna(0)
avg_dep_delays = data_of_interest.groupby('date').mean().reset_index() #Calculate mean departure delay for each date, this also makes date a column again instead of the index
index_of_interest = avg_dep_delays['dep_delay'].idxmax()
largest_avg_dep_delay_row = avg_dep_delays.loc[index_of_interest]
largest_avg_dep_delay_date = largest_avg_dep_delay_row['date'].strftime('%Y/%-m/%-d')
largest_avg_dep_delay = round(largest_avg_dep_delay_row['dep_delay'], 2)
print(f"\nQuestion 5: Date with largest average departure delay: {largest_avg_dep_delay_date} with delay of {largest_avg_dep_delay} mins.")

#Question 6 Which date has the largest average arrival delay? pd slice with date and float
percent_missing_arr_delay_data = check_missing_data(flights_data, 'arr_delay')
print(f"\n\tQ6 data quality check - % missing arrival delay data: {percent_missing_arr_delay_data}%") #This is just a data quality check!
# Since the missing arrival delay data is small (2.8%), I will assume the delay is 0 for those entries.
arr_delay_data = flights_data[['date', 'arr_delay']].fillna(0)
avg_arr_delays = arr_delay_data.groupby('date').mean().reset_index() 
index = avg_arr_delays['arr_delay'].idxmax()
largest_avg_arr_delay_row = avg_arr_delays.loc[index]
largest_avg_arr_delay_date = largest_avg_arr_delay_row['date'].strftime('%Y/%-m/%-d')
largest_avg_arr_delay = round(largest_avg_arr_delay_row['arr_delay'], 2)
print(f"\nQuestion 6: Date with largest average arrival delay: {largest_avg_arr_delay_date} with delay of {largest_avg_arr_delay} mins.")

#Question 7 Which flight departing LGA or JFK in 2013 flew the fastest? pd slice with tailnumber and speed (speed = distance / airtime)
flights_NYC_2013 = flights_data[(flights_data['year'] == 2013) & (flights_data['origin']).isin(['LGA', 'JFK'])] #
percent_missing_flight_time = check_missing_data(flights_NYC_2013, 'air_time')
percent_missing_flight_distance = check_missing_data(flights_NYC_2013, 'distance')

#Data quality check for missing air time and distance data
print(f"\n\tQ7 data quality check - % missing air time data: {percent_missing_flight_time}%") #2.65% missing
print(f"\n\tQ7 data quality check - % missing distance data: {percent_missing_flight_distance}%") #0.0% missing
# Since there is a small amount of missing air time data (2.65%), I will drop those rows so that I can calculate speed where there is complete data.
flights_NYC_2013_cleaned = flights_NYC_2013.dropna(subset=['air_time'])

flights_NYC_2013_cleaned['speed'] = round(flights_NYC_2013_cleaned['distance'] / (flights_NYC_2013_cleaned['air_time'] / 60), 2) #Made new column for the calculated speed in miles per hour (mph)
speed_data = flights_NYC_2013_cleaned[['tailnum', 'speed', 'origin']]
# index_of_fastest_flight = speed_data['speed'].idxmax()
fastest_flight_row = speed_data.loc[speed_data['speed'].idxmax()]
print(f"\nThe flight with tail number {fastest_flight_row['tailnum']}, departed from {fastest_flight_row['origin']}, and flew the fastest with a speed of {fastest_flight_row['speed']} mph.\n")

#Question 8 Replace all NaN values in the weather pd dataframe with 0s. Pd with no NaNs.
weather_data_cleaned = weather_data.fillna(0)
print("\n", weather_data_cleaned.isnull().sum() == 0) #Checking that all NaN values were replaced with 0s

#Question 9 How many observation were made in February? int
february_weather_data = weather_data_cleaned[weather_data_cleaned['month'] == 2.0]
february_observations = len(february_weather_data)
print(f"\nQuestion 9: There were {february_observations} weather observations in February.")

#Question 10 What was the mean for humidity in February? float
# Note: Since all NaN values were replaced with 0s, and humidity could technically be 0%, I am going to maintain the 0s in the data (if they exist in humidity column)
mean_february_humidity = round(february_weather_data['humid'].mean(), 2)
print(f"\nQuestion 10: The mean February humidity was {mean_february_humidity}%.")

#Question 11 What was the standard deviation for humidity in February? float
std_february_humidity = round(february_weather_data['humid'].std(), 2)
print(f"\nQuestion 11: The standard deviation for February humidity was {std_february_humidity}%.")