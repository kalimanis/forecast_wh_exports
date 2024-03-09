import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib  # Import joblib for model saving

# Load the enhanced data
file_path = 'enhanced_casesfor.xlsx'
data = pd.read_excel(file_path, index_col='Date')

# Prepare features and target variable
X = data.drop(['PickingCases'], axis=1)
y = data['PickingCases']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set and evaluate
predictions = rf.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Save your model
model_file_path = 'random_forest_model.joblib'
joblib.dump(rf, model_file_path)
print(f'Model saved to {model_file_path}')
