import pandas as pd
import numpy as np

# Question 1: Compute the euclidean distance between series (points) p and q, without using a packaged formula.

p = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

squared_difference = 0

for i in range(len(p)):
    squared_difference += (q[i] - p[i]) ** 2

euclidean_distance = round(float(np.sqrt(squared_difference)), 4)
print(f"\nQuestion 1: The euclidean distance between p and q = {euclidean_distance}")


# Question 2: Change the order of columns of a dataframe. Interchange columns 'a' and 'c'.

df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))
df_reordered = df[['c', 'b', 'a', 'd', 'e']]
print(f"\nQuestion 2: Reordered dataframe = \n{df_reordered}")

# Question 3: Change the order of columns of a dataframe. 
# Create a generic function to interchange two columns, without hardcoding column names.

df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))

def interchange_columns(dataframe, col1_name, col2_name):
    columns = list(dataframe.columns)
    col1_index = columns.index(col1_name)
    col2_index = columns.index(col2_name)

    columns[col1_index] = col2_name
    columns[col2_index] = col1_name

    df = dataframe[columns]
    return df

new_df = interchange_columns(df, 'a', 'e')
print(f"\nQuestion 3: Example using interchange_columns function to switch columns 'a' and 'e':\n{new_df}")


# Question 4: Format or suppress scientific notations in a pandas dataframe. 
# Suppress scientific notations like ‘e-03’ in df and print up to 4 numbers after decimal.

df4 = pd.DataFrame(np.random.random(4)**10, columns=['random'])
print(f"\nQ4 original dataframe: \n{df4}")
pd.set_option('display.float_format', '{:.4f}'.format) #This sets the float fromat for pandas to display up to 4 decimal places
print(f"\nQuestion 4: dataframe with no scientific notation: \n{df4}")

# Question 5: Create a new column that contains the row number of nearest column by euclidean distance. 
# Create a new column such that, each row contains the row number of nearest row-record by euclidean distance.

df5 = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1), columns=list('pqrs'), index=list('abcdefghij'))
print(f"\n\tQ5 - Original dataframe: \n{df5}") #This was just printed for error checking

def find_euclidean_distance(row1, row2):
    '''
    This function calculates the euclidean distance between two pandas series (rows from a dataframe).
    Inputs:
        row1: pandas series
        row2: pandas series
    Output: (float) euclidean distance rounded to 4 decimal places
    '''
    squared_difference = 0
    for column in row1.index:
        squared_difference += (row1[column] - row2[column]) ** 2
    euclidean_distance = float(np.sqrt(squared_difference))
    return round(euclidean_distance, 4)

def create_row_distance_df(dataframe):
    '''
    This function creates an nxn dataframe containing the euclidean distances between each pair of rows in the input dataframe.
    This is accomplished by iterating through each rows pairing and creating a dictionary of distances, which is converted to a dataframe.
    Inputs:
        dataframe: pandas dataframe
    Output: pandas dataframe containing euclidean distances between rows. 
        The diagonal is filled with NaN since the row being compared to itself is not needed.
    '''
    distances = {}
    for row1_name in dataframe.index:
        distances[row1_name] = {}
        row1 = dataframe.loc[row1_name]
        for row2_name in dataframe.index:
            if row1_name != row2_name:
                row2 = dataframe.loc[row2_name]
                euclidean_distance = find_euclidean_distance(row1, row2)
                distances[row1_name][row2_name] = euclidean_distance
            else:
                distances[row1_name][row2_name] = np.nan
    
    return pd.DataFrame.from_dict(distances)

row_dist_df = create_row_distance_df(df5)

for index, row in df5.iterrows():
    nearest_row_index = row_dist_df.loc[index].idxmin()
    df5.at[index, 'nearest_row'] = nearest_row_index
    df5.at[index, 'dist'] = row_dist_df.loc[index].min()

print(f"\n\tQ5 - row distance df generated: \n{row_dist_df}") #This was just printed for error checking
print(f"\nQuestion 5: Updated dataframe with nearest row and euclidean distance columns: \n{df5}")

# Question 6: Correlation is a statistical technique that shows how two variables are related. Pandas dataframe.corr() method is
# used for creating the correlation matrix. It is used to find the pairwise correlation of all columns in the dataframe.
# Any na values are automatically excluded. For any non-numeric data type columns in the dataframe it is ignored.

data = {'A': [45, 37, 0, 42, 50],
        'B': [38, 31, 1, 26, 90],
        'C': [10, 15, -10, 17, 100],
        'D': [60, 99, 15, 23, 56],
        'E': [76, 98, -0.03, 78, 90]
         }

df6 = pd.DataFrame(data)
correlation_matrix = df6.corr()
print(f"\nQuestion 6: Correlation matrix for the given dataframe: \n{correlation_matrix}")