import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
import os

# File path
file_path = r'C:\Users\S K Mohanty\OneDrive\Desktop\Flight Phase Identification\fdr1.csv'

# Check if file exists
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Try to load the dataset
try:
    data = pd.read_csv(file_path)
    print("File loaded successfully.")
except PermissionError as e:
    raise PermissionError(f"Permission denied: {e}")
except FileNotFoundError as e:
    raise FileNotFoundError(f"File not found: {e}")

# Display the first few rows of the dataframe
print(data.head())

# Extract features
features = data[['Time', 'Speed', 'Altitude']]

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the Elbow Method
def elbow_method(normalized_features):
    inertia = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalized_features)
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()

elbow_method(normalized_features)

# Ask the user for the number of clusters
n_clusters = int(input("Enter the number of clusters: "))

# Perform KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(normalized_features)

# Add the cluster labels to the original data
data['Cluster'] = clusters

# Ask the user for cluster names
cluster_names = {}
for i in range(n_clusters):
    name = input(f"Enter the name for cluster {i}: ")
    cluster_names[i] = name

# Map the cluster names to the data
data['Cluster Name'] = data['Cluster'].map(cluster_names)

# Visualize the clusters in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data['Time'], data['Speed'], data['Altitude'], c=data['Cluster'], cmap='viridis')

ax.set_xlabel('Time')
ax.set_ylabel('Speed')
ax.set_zlabel('Altitude')
plt.title('3D Visualization of Clusters')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()

# Evaluate clustering with Silhouette Score
score = silhouette_score(normalized_features, clusters)
print(f'Silhouette Score: {score}')

# Optional: If you want to use LSTM for temporal analysis
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore # type: ignore

# Prepare the dataset for LSTM
data['Cluster'] = data['Cluster'].astype(str)  # Convert clusters to string for one-hot encoding
data = pd.get_dummies(data, columns=['Cluster'])

X = data[['Time', 'Speed', 'Altitude']].values
y = data.drop(['Time', 'Speed', 'Altitude'], axis=1).values

X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for LSTM

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Visualize training history
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
