import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway, ttest_ind

# Load the dataset
df = pd.read_excel(r'/Users/andeysaikiran/Downloads/Boston Crime Dataset_Final_Final.xlsx') 
df.info()  # Display dataset information
print(df.head())  # Display first few rows


# Drop unnecessary columns
df = df.drop(['Lat', 'Long'], axis=1)

# Convert 'OCCURRED_ON_DATE' to datetime format
df['OCCURRED_ON_DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE'])

# Fill missing values
df['SHOOTING'] = df['SHOOTING'].fillna('No')  # Assuming 'No' means no shooting occurred
df['STREET'] = df['STREET'].fillna('Unknown')

# Rename columns for clarity
df.rename(columns={'Sno': 'Serial_Number'}, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Standardize text data
df['DISTRICT'] = df['DISTRICT'].str.title()
df['DAY_OF_WEEK'] = df['DAY_OF_WEEK'].str.title()
df['Season'] = df['Season'].str.title()
df['OFFENSE_DESCRIPTION'] = df['OFFENSE_DESCRIPTION'].str.title()

# Outlier detection and removal
q_low = df['Per Capita Income'].quantile(0.01)
q_high = df['Per Capita Income'].quantile(0.99)
df = df[(df['Per Capita Income'] > q_low) & (df['Per Capita Income'] < q_high)]

# Additional error checking for non-negative values
columns_to_check = ['Total Population', 'Per Capita Income', 'Total in Poverty', 'Total Educated', 'Police Force Available']
for column in columns_to_check:
    df[column] = df[column].apply(lambda x: max(x, 0))


# Distribution of Crimes by District
plt.figure(figsize=(10, 6))
district_counts = df['DISTRICT'].value_counts()
sns.barplot(x=district_counts.index, y=district_counts.values, palette='coolwarm')
plt.title('Distribution of Crimes by District')
plt.xlabel('District')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Crime Severity Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Severity'], bins=30, kde=True, color='blue')
plt.title('Distribution of Crime Severity')
plt.xlabel('Severity')
plt.ylabel('Frequency')
plt.show()

# Trends Over Time (Yearly and Monthly)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.lineplot(data=df, x='YEAR', y='Severity', estimator='mean', errorbar=None, marker='o')
plt.title('Yearly Trend of Crime Severity')
plt.xlabel('Year')
plt.ylabel('Average Severity')

plt.subplot(1, 2, 2)
monthly_severity = df.groupby(['YEAR', 'MONTH'])['Severity'].mean().reset_index()
sns.lineplot(data=monthly_severity, x='MONTH', y='Severity', hue='YEAR', palette='viridis', estimator=None, legend='full')
plt.title('Monthly Trends in Crime Severity')
plt.xlabel('Month')
plt.ylabel('Average Severity')
plt.tight_layout()
plt.show()


# Hypothesis 1: Crime Severity by Season
seasonal_data = df.groupby('Season')['Severity'].mean().reset_index()
season_groups = [df['Severity'][df['Season'] == season] for season in df['Season'].unique()]
anova_result = f_oneway(*season_groups)

# Hypothesis 2: Crime Severity by Income Level
median_income = df['Per Capita Income'].median()
df['Income Level'] = np.where(df['Per Capita Income'] >= median_income, 'High Income', 'Low Income')
income_data = df.groupby('Income Level')['Severity'].mean().reset_index()
high_income_severity = df['Severity'][df['Income Level'] == 'High Income']
low_income_severity = df['Severity'][df['Income Level'] == 'Low Income']
t_test_result = ttest_ind(high_income_severity, low_income_severity)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Season', y='Severity', data=seasonal_data)
plt.title('Mean Crime Severity by Season')
plt.ylabel('Mean Severity')
plt.xlabel('Season')

plt.subplot(1, 2, 2)
sns.barplot(x='Income Level', y='Severity', data=income_data)
plt.title('Mean Crime Severity by Income Level')
plt.ylabel('Mean Severity')
plt.xlabel('Income Level')
plt.tight_layout()
plt.show()

print("ANOVA test result for Seasonal Crime Severity: F-statistic = {:.2f}, p-value = {:.3f}".format(anova_result.statistic, anova_result.pvalue))
print("T-test result for Income Level Crime Severity: t-statistic = {:.2f}, p-value = {:.3f}".format(t_test_result.statistic, t_test_result.pvalue))




# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Remove highly correlated features
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
df_reduced = df.drop(columns=to_drop)

# Prepare data for training
X = df_reduced.drop('Severity', axis=1)
y = df_reduced['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()



# ----

# Assuming 'OCCURRED_ON_DATE' is already converted to datetime and set as index if not:
df['OCCURRED_ON_DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE'])
df.set_index('OCCURRED_ON_DATE', inplace=True)

# Resample data to get annual crime counts
yearly_crimes = df['Serial_Number'].resample('A').count()  # 'A' stands for annual frequency


from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Fit the ARIMA model
model = ARIMA(yearly_crimes, order=(1, 1, 1))  # Adjust the order based on your specific dataset
fitted_model = model.fit()

# Forecast the next 5 years
forecast = fitted_model.forecast(steps=5)  # Forecast the next 5 years

# Plot the historical data and the forecast
plt.figure(figsize=(10, 5))
plt.plot(yearly_crimes, label='Historical Annual Crime Count')
plt.plot(forecast, label='Forecasted Crime Count', color='red')
plt.title('Annual Crime Count Forecast')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.legend()
plt.show()


#---- 

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel(r'/Users/andeysaikiran/Downloads/Boston Crime Dataset_Final_Final.xlsx')

# Convert 'OCCURRED_ON_DATE' to datetime format without timezone
df['OCCURRED_ON_DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE']).dt.tz_localize(None)

# Set 'OCCURRED_ON_DATE' as the DataFrame index for time series analysis
df.set_index('OCCURRED_ON_DATE', inplace=True)

# Aggregate data monthly and prepare for Prophet
monthly_crimes = df.resample('M').size().reset_index()
monthly_crimes.columns = ['ds', 'y']  # Prophet requires the column names to be 'ds' and 'y'

# Initialize and fit the Prophet model
m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
m.fit(monthly_crimes)

# Make a dataframe to hold predictions for the next 2 years
future = m.make_future_dataframe(periods=24, freq='M')

# Forecast future data
forecast = m.predict(future)

# Plot the forecast
fig = m.plot(forecast)
plt.title('Crime Rate Forecast for Next 2 Years')
plt.xlabel('Date')
plt.ylabel('Crime Count')
plt.show()

# Plot the components of the forecast
fig2 = m.plot_components(forecast)

