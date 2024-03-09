import pandas as pd
from datetime import timedelta
import joblib

# Load the saved model
model_file_path = 'random_forest_model.joblib'
rf = joblib.load(model_file_path)

# Assuming last part of enhanced data is similar in feature preparation
data = pd.read_excel('enhanced_casesfor.xlsx', index_col='Date')  # For feature structure consistency

# Get the feature names used in the training data
feature_names = data.columns.drop(['PickingCases'])

# Prepare the data for the next 10 working days
last_date = data.index.max()
new_dates = pd.date_range(start=last_date + timedelta(days=1), periods=10, freq='B')
new_data = pd.DataFrame(index=new_dates)
new_data['Weekday'] = new_data.index.day_name()

# Ensure all features match the training features, particularly weekdays
# Since we used get_dummies, reapply it here to ensure alignment
new_data = pd.get_dummies(new_data, columns=['Weekday'])
new_data = new_data.reindex(columns=feature_names, fill_value=0)  # Fill missing columns with 0s

# Forecasting with the loaded model
forecasts = rf.predict(new_data)
forecast_series = pd.Series(data=forecasts, index=new_dates)
print("Forecasts for the next 10 working days:")
print(forecast_series)
