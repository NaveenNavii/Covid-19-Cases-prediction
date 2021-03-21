import numpy as np
import pandas as pd
from sklearn import linear_model
import  datetime
import matplotlib.pyplot as plt

startdate = "17/07/2020"
print(f"Last covid-19 data: 17/17/20\nDeaths: {4971}\nConfirmed: {173763}\n")

d = pd.read_csv('covid_dataset.csv', usecols=['Date', 'Deaths', 'Confirmed'])
df = d.groupby('Date').sum()
indexNames = df[df['Confirmed'] <= 35000].index
df.drop(indexNames, inplace=True)
indexNames = df[df['Confirmed'] >= 175000].index
df.drop(indexNames, inplace=True)
df['Sno'] = np.arange(1, len(df) + 1)
df = df[['Sno', 'Deaths', 'Confirmed']]

x = df[['Sno']]
y = df[['Confirmed']]
z = df[['Deaths']]

confirmed = linear_model.LinearRegression()
deaths = linear_model.LinearRegression()
confirmed.fit(x, y)
deaths.fit(x, z)
m1 = confirmed.coef_[0][0]
c1 = confirmed.intercept_[0]
m2 = deaths.coef_[0][0]
c2 = deaths.intercept_[0]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter('Sno', 'Confirmed', data=df, c='g', label='Actual')
plt.plot(x, m1 * x + c1, label='Predicted')
plt.grid(True)

ax.scatter('Sno', 'Deaths', data=df, c='b', label='Actual')
plt.plot(x, m2 * x + c2, label='Predicted')
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Confirmed & Deaths')
plt.title('Covid-19')
plt.legend()
plt.show()

a = input('How many days after have to predict confirmed rate: ')
for i in range(1, int(a)+1):
    confirmed_result = confirmed.predict([[(30+i)]])
    deaths_result = deaths.predict([[(30 + i)]])
    end_date = pd.to_datetime(startdate) + pd.DateOffset(days=i)
    end_date = end_date.strftime('%d/%m/%Y')
    print(f"\nDate: {end_date}, confirmed rate is {round(confirmed_result[0][0])}")
    print(f"Date: {end_date}, death rate is {round(deaths_result[0][0])}")
