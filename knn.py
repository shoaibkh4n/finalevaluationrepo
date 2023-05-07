from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Define the factors that will be used to determine similarity between locations
factors = ['distance', 'accessibility', 'user_preferences', 'weather']

# Collect data on the different locations within the indoor environment
data = [
    {'location': 'A', 'distance': 10, 'accessibility': 'wheelchair', 'user_preferences': 'elevator', 'weather': 'sunny'},
    {'location': 'B', 'distance': 20, 'accessibility': 'stairs', 'user_preferences': 'stairs', 'weather': 'cloudy'},
    {'location': 'C', 'distance': 30, 'accessibility': 'elevator', 'user_preferences': 'stairs', 'weather': 'rainy'},
]
le1= LabelEncoder()
# Preprocess the data to ensure that it is in a suitable format for use with the KNN algorithm
X = [[d[f] for f in factors] for d in data]
data_encoded = np.apply_along_axis(LabelEncoder().fit_transform, axis=0, arr=X)
X=data_encoded

le = LabelEncoder()

y = [d['location'] for d in data]
y=le.fit_transform(y)

# Implement the KNN algorithm
k = 3 # choose the number of neighbors to consider
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

knn.fit(X, y)

current_location = {'distance': 15, 'accessibility': 0, 'user_preferences': 1, 'weather': 2}
input_data = [[current_location[f] for f in factors]]
encoded_input = knn.predict(input_data)
recommendations = le.inverse_transform(encoded_input)
print(recommendations)  # output: ['A', 'B']
