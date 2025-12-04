import numpy as np
import matplotlib.pyplot as plt


from glob import glob
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from numpy import asarray
from sklearn.preprocessing import PolynomialFeatures

# 1. Load Excel file
files = glob("Data/*xlsx") # adjust path if needed
files_hourly = glob("Data_monthly/*xlsx")
df_list = []
for f in files:
    df = pd.read_excel(f, sheet_name = None)
    print(f"File: {f}")
    # Combine all sheets, adding the sheet name as new feature 'Zone'
    for zone_name, data in df.items():
        data = data.copy()
        data['Zone'] = zone_name
        df_list.append(data)
for f in files_hourly:
    df = pd.read_excel(f, sheet_name = None)
    print(f"File: {f}")
    # Combine all sheets, adding the sheet name as new feature 'Zone'
    for zone_name, data in df.items():
        data = data.copy()
        df_list.append(data)
# Merge all zones into one dataframe
df = pd.concat(df_list, ignore_index=True)



# 2. Select relevant columns
features = ['Year', 'Month', 'Energy', 'CDD', 'HDD', 'Peak_Day', 
            'Peak_Demand', 'Peak_Hour', 'Peak_DB', 'Peak_DP', 'Zone']

features_hourly = ['Date', 'Hr_End', 'DA_Demand', 'RT_Demand', 'RT_LMP', 'RT_EC', 'RT_CC', 'RT_MLC', 'Dry_Bulb', 'Dew_Point']

df_monthly = df[features]

df_hourly = df[features_hourly]

def predict(df, input):
    # 3. Handle missing values (drop rows with NaN)
    df = df.dropna()

    # 4. Encode categorical column 'Zone' if it's not numeric
    if input == True:
        df = pd.get_dummies(df, columns=['Zone'], drop_first=True)

        # 5. Separate target and predictors
        target = 'Peak_Demand' 
       
    else:
        target = 'RT_Demand'
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Hr_End'] = pd.to_numeric(df['Hr_End'], errors='coerce')

        df = df.dropna(subset=['Date', 'Hr_End'])

        df['Year']  = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day']   = df['Date'].dt.day

        df = df.drop(columns=['Date'])
        # cols = ['Hr_End']
        # data.loc[:, cols] = data[cols].astype(int)


        # # Create the DateTime Index
        # def create_datetime(row):

        #     dt = pd.to_datetime(row['Date']) + pd.Timedelta(hours=row['Hr_End'])
        #     # Correct for hour ending 24
        #     if row['Hr_End'] == 24:
        #         return dt - pd.Timedelta(hours=1)
        #     return dt


        # data.loc[:, 'DateTime'] = data.apply(create_datetime, axis=1)
    X = df.drop(columns=[target])
    y = df[target]
    # 6. Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Initialize and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 8. Make predictions
    y_pred = model.predict(X_test)

    # 9. Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")

    # 10. Optionally, show sample predictions
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'error': y_test - y_pred})
    print(results.head(10))

predict(df_monthly, True)

predict(df_hourly, False)

# cdd = df[['CDD']]
# Peak_Hours = df[['Peak_Hour']]
# poly = PolynomialFeatures(degree = 2, include_bias=False)
# cdd_trans = poly.fit_transform(cdd)

# Hours_trans = poly.fit_transform(Peak_Hours)

# X1 = df.drop(columns=[target, 'CDD', 'Peak_Hour', 'Peak_DB', 'Peak_DP'])



# cdd_poly_df = pd.DataFrame(cdd_trans, columns = ['CDD', 'CDD^2'], index = df.index)

# hours_poly_df = pd.DataFrame(Hours_trans, columns = ['Hours', 'Hours^2'], index = df.index)

# X1 = pd.concat([X1, cdd_poly_df], axis = 1)
# X1 = pd.concat([X1, hours_poly_df], axis = 1)
# y1 = df[target]

# X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

# model1 = LinearRegression()
# model1.fit(X_train1, y_train1)

# y_pred1 = model1.predict(X_test1)

# mse1 = mean_squared_error(y_test1, y_pred1)
# r2_1 = r2_score(y_test1, y_pred1)

# print(f"Mean Squared Error: {mse1:.2f}")
# print(f"R² Score: {r2_1:.3f}")

# results1 = pd.DataFrame({'Actual': y_test1, 'Predicted': y_pred1, 'error': y_test1 - y_pred1})
# print(results1.head(10))
