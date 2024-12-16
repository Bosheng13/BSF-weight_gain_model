import pandas as pd
from sklearn.preprocessing import StandardScaler
data = pd.read_excel('d:/Weightgain.xlsx')
data
data.info()

########Check for missing values#########
print(data.isnull().sum())



import matplotlib.pyplot as plt
import seaborn as sns

###### Plot histograms for each feature#########
data.hist(bins=10, figsize=(12, 10))
plt.tight_layout()
plt.show()

#####Show scatter plots to explore relationships between features and target variable#######
sns.pairplot(data, x_vars=['C', 'N', 'Vb', 'Vc', 'Fat', 'cellulose'], y_vars=['Weightgain'])
plt.show()

####Select features and target variable####
X = data[['C', 'N', 'Vb', 'Vc', 'Fat', 'cellulose']]
y = data['Weightgain']


######Standardize the features#######
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:5])

#######Create a new DataFrame with the scaled features and the target variable######
scaled_data = pd.DataFrame(X_scaled, columns=['C', 'N', 'Vb', 'Vc', 'Fat', 'cellulose'])
scaled_data['Weightgain'] = y


########Save to an Excel file########
scaled_data.to_excel('d:/scaled-weight.xlsx', index=False)
print("Scaled data has been saved to 'scaled-weight.xlsx'")
print(scaled_data.head())