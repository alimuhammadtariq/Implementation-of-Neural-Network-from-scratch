import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Reading the data from the game
data = pd.read_csv('game_data.csv', names=['X1', 'X2', 'Y1', 'Y2'])

# Removing all the null values
data = data.dropna()

# Removing duplicates
data = data.drop_duplicates(keep='first')

# Splitting X and Y data
X = data[['X1', 'X2']]
Y = data[['Y1', 'Y2']]

# Scaling the data between 0 and 1 using MinMaxScaler
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# Creating new DataFrames with scaled values
scaled_X = pd.DataFrame(X_scaled, columns=['X1', 'X2'])
scaled_Y = pd.DataFrame(Y_scaled, columns=['Y1', 'Y2'])

# Save the MinMaxScaler objects to files
joblib.dump(scaler_X, 'minmax_scaler_X.joblib')
joblib.dump(scaler_Y, 'minmax_scaler_Y.joblib')

# Splitting data between training, testing, and validation
X_train, X_validate_test, Y_train, Y_validate_test = train_test_split(scaled_X, scaled_Y, train_size=0.7)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_validate_test, Y_validate_test, test_size=0.5)

# Writing data to CSV files
X_train.to_csv('X_train.csv', index=False)
Y_train.to_csv('Y_train.csv', index=False)
X_validate.to_csv('X_validate.csv', index=False)
Y_validate.to_csv('Y_validate.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
Y_test.to_csv('Y_test.csv', index=False)
