# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# %%
customer = pd.read_csv('Ecommerce Customers.csv')
# %%
customer.head()
# %%
customer.info()
# %%
customer.describe()
# %%
customer.drop(columns=['Email', 'Address', 'Avatar'], inplace=True)
# %%
plt.figure(figsize=(12, 6))
sns.scatterplot(x=customer['Yearly Amount Spent'], y=customer['Length of Membership'])
# %%
y = customer['Yearly Amount Spent']
# %%
customer.drop(columns=['Yearly Amount Spent'], inplace=True)
# %%
x = customer.copy()
# %%
x.head()
# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
# %%
lr_model = LinearRegression()
# %%
lr_model.fit(x_train, y_train)
# %%
y_pred = lr_model.predict(x_test)
# %%
plt.scatter(y_pred, y_test)
# %%
mae = mean_absolute_error(y_pred, y_test)
mae
# %%
mse = mean_squared_error(y_pred, y_test)
mse
# %%
r2 = r2_score(y_pred, y_test)
r2
# %%
res = sns.distplot(x)
plt.show()
# %%
