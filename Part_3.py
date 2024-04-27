import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:\\Kode\\GitHub\\Part E\\avocado.csv')

# Convert 'Date' to datetime and extract year, month, and day
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data.drop('Date', axis=1, inplace=True)  # Drop the original 'Date' column if no longer needed

# Handling categorical data with one-hot encoding
encoder = OneHotEncoder()
regions_encoded = encoder.fit_transform(data[['region']]).toarray()  # Convert sparse matrix to a dense matrix
regions_encoded_df = pd.DataFrame(regions_encoded, columns=encoder.get_feature_names_out(['region']))

# Concatenate encoded data with the original dataframe
data = pd.concat([data.drop(['region'], axis=1), regions_encoded_df], axis=1)

# Prepare data for clustering
features = data.select_dtypes(include=[np.number])  # Selecting only numeric data for clustering
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Hierarchical Clustering
# Generate the linkage matrix
Z = linkage(features_scaled, 'complete')

# Plot dendrogram
plt.figure(figsize=(10, 7))
plt.title('Hierarchical Clustering Dendrogram')
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()

# K-Means Clustering
# Choose the number of clusters
k = 5
kmeans = KMeans(n_clusters=k, random_state=42).fit(features_scaled)
labels = kmeans.labels_

# Visualizing the clusters
plt.figure(figsize=(10, 7))
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=labels, cmap='rainbow')
plt.title('K-Means Clustering')
plt.show()

# Calculate Silhouette Score for K-Means
score = silhouette_score(features_scaled, labels)
print('Silhouette Score for K-Means with k = {}: {:.3f}'.format(k, score))

