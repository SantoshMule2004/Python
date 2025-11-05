# EXERCISE 1: Data Cleaning Challenge
# Using your BMW dataset, try to:

import pandas as pd
dfBmw = pd.read_csv("./bmw_with_missing_data.csv")


# 1. Find how many missing values are in each column
missingValues = dfBmw.isnull().sum()
print("Missing values in each column:")
print(missingValues)

# 2. Remove all rows with missing prices 
dfBmw['price'] = pd.to_numeric(dfBmw['price'], errors='coerce')
missingPrices = dfBmw.dropna(subset=['price'])
print("removed all rows with missing prices:")
print(missingPrices)

# 3. Fill missing years with the most common year
commonYear = dfBmw['year'].mode()[0]
dfBmw['year'].fillna(value=commonYear, inplace=True)
print(dfBmw.head())

# 4. Find and remove duplicate entries
droppedDuplicates = dfBmw.drop_duplicates()
print(droppedDuplicates)

# 5. Calculate average price after cleaning
avg_price = dfBmw['price'].mean()
dfBmw['price'] = dfBmw['price'].fillna(avg_price)
print("Average Price:", dfBmw)