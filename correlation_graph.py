import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
avocado_data = pd.read_csv('C:\\Kode\\GitHub\\Part E\\avocado.csv')  # Update the path to where your dataset is stored

# Remove Unnamed: 0 column
avocado_data_cleaned = avocado_data.drop(columns=['Unnamed: 0'])

# Save the cleaned data 
avocado_data_cleaned.to_csv('C:\\Kode\\GitHub\\Part E\\avocado.csv', index=False)

# Calculate the correlation matrix 
correlation_matrix = avocado_data_cleaned.select_dtypes(include=['float64', 'int64']).corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title('Correlation Matrix of Numeric Features')
plt.show()
