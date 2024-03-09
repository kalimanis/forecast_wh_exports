import pandas as pd

# Load the dataset
file_path = 'cases.xlsx'  # Replace with your actual data file path
data = pd.read_excel(file_path)

# Convert 'Date' into datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Create additional time features
data['Weekday'] = data.index.day_name()
data['Is_Friday'] = (data['Weekday'] == 'Friday').astype(int)
data['Is_EndOfMonth'] = (data.index.is_month_end).astype(int)
data['Day_of_Month'] = data.index.day

# One-hot encode weekdays
data = pd.get_dummies(data, columns=['Weekday'], drop_first=True)

# Save the enhanced data for modeling
enhanced_file_path = 'enhanced_casesfor.xlsx'
data.to_excel(enhanced_file_path)
print('Data preparation and feature engineering complete.')
