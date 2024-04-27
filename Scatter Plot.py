import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
avocado_data = pd.read_csv('C:\\Kode\\GitHub\\Part E\\avocado.csv')  # Update the path to where your dataset is stored

# Standardizing the data (important for PCA)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(avocado_data[['AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags']])

# PCA transformation
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled)
principal_df = pd.DataFrame(data = principal_components, columns = ['Principal Component 1', 'Principal Component 2'])

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=avocado_data, x='AveragePrice', y='Total Volume', hue='type', style='type', alpha=0.5)
plt.title('Scatter Plot of Average Price vs Total Volume by Type')
plt.xlabel('Average Price')
plt.ylabel('Total Volume')
plt.yscale('log') 
plt.show()