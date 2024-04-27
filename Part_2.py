import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('C:\\Kode\\GitHub\\Part E\\avocado.csv')

# Convert Date to datetime and extract year, month, and day
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data.drop('Date', axis=1, inplace=True)  

# Handling categorical data 
encoder = OneHotEncoder()
regions_encoded = encoder.fit_transform(data[['region']]).toarray()  # Convert sparse matrix to a dense matrix
regions_encoded_df = pd.DataFrame(regions_encoded, columns=encoder.get_feature_names_out(['region']))

# Concatenate encoded data
data = pd.concat([data.drop(['region'], axis=1), regions_encoded_df], axis=1)

# Encode the target variable
label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])

# Split data into features and target
X = data.drop(['type'], axis=1)
y = data['type']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForest Classifier
clf_rf = RandomForestClassifier(n_estimators=100)
clf_rf.fit(X_train, y_train)

# Predictions with RandomForest
predictions_rf = clf_rf.predict(X_test)

# Evaluation of RandomForest
print("Random Forest Classifier Accuracy:", accuracy_score(y_test, predictions_rf))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, predictions_rf))

# Initialize and train the Decision Tree Classifier
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, y_train)

# Predictions with Decision Tree
predictions_dt = clf_dt.predict(X_test)

# Evaluation of Decision Tree
print("Decision Tree Classifier Accuracy:", accuracy_score(y_test, predictions_dt))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, predictions_dt))

# Calculate the total number of entries in the dataset
total_entries = len(data)

# Calculate the number of entries in the training and testing datasets
train_count = len(X_train)
test_count = len(X_test)

# Calculate the percentage of entries in the training and testing datasets
train_percentage = (train_count / total_entries) * 100
test_percentage = (test_count / total_entries) * 100

# Output the results
print(f"Total number of data objects: {total_entries}")
print(f"Number of training data objects: {train_count} ({train_percentage:.2f}%)")
print(f"Number of test data objects: {test_count} ({test_percentage:.2f}%)")
