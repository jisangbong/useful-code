import pandas as pd
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

# Sample first and last names
first_names = ['John', 'Emma', 'Michael', 'Olivia', 'David', 'Sophia', 'Daniel', 'Isabella', 'James', 'Mia']
last_names = ['Smith', 'Johnson', 'Brown', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin']

# Settings
num_rows = 20
company_choices = ['Apple', 'MS', 'Google']
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 12, 31)

# Function to generate a random datetime between two dates
def random_datetime(start, end):
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)

# Function to calculate full elapsed time with hours, minutes, seconds
def elapsed_full(start_date, end_date):
    delta = relativedelta(end_date, start_date)
    total_seconds = int((end_date - start_date).total_seconds())
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{delta.years}y {delta.months}m {delta.days}d {hours}h {minutes}m {seconds}s"

# Generate random data
company_names = [random.choice(company_choices) for _ in range(num_rows)]
names = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(num_rows)]
employment_dates = [random_datetime(start_date, end_date) for _ in range(num_rows)]

# Create the DataFrame
df = pd.DataFrame({
    'Company': company_names,
    'Name': names,
    'Date of Employment': employment_dates
})

# 1000 days after employment
df['1000_days'] = df['Date of Employment'] + pd.to_timedelta(1000, unit='D')

# 1000 business days after employment
df['1000_bizdays'] = df['Date of Employment'] + pd.offsets.BDay(1000)

# Today's date
today = datetime.today().replace(microsecond=0)
df['today'] = today

# Calendar day difference
df['todaydays'] = (today - df['Date of Employment']).dt.days

# Business day difference
df['todaybizdays'] = df['Date of Employment'].apply(lambda x: np.busday_count(x.date(), today.date()))

# Elapsed time in years, months, days, hours, minutes, seconds
df['today_elapsed'] = df['Date of Employment'].apply(lambda x: elapsed_full(x, today))

# Format the date columns
df['Date of Employment'] = df['Date of Employment'].dt.strftime('%m/%d/%Y %I:%M:%S %p')
df['1000_days'] = pd.to_datetime(df['1000_days']).dt.strftime('%m/%d/%Y %I:%M:%S %p')
df['1000_bizdays'] = pd.to_datetime(df['1000_bizdays']).dt.strftime('%m/%d/%Y %I:%M:%S %p')
df['today'] = df['today'].dt.strftime('%m/%d/%Y %I:%M:%S %p')

# Display the result
print(df.to_string())