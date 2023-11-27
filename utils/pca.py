from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Assume X is your data matrix with observations as rows and features as columns
# For demonstration, let's create a synthetic dataset with 10 observations and 5 features.
np.random.seed(0)  # Seed for reproducibility
X = np.random.rand(10, 2)
plt.plot(X)
plt.show()

print(X)

# Standardize the features (mean = 0 and variance = 1)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Perform PCA to reduce the data to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# The transformed data (in the new PCA space)
print("Transformed Data:\n", X_pca)

# The amount of variance explained by each of the selected components
print("Explained variance ratio:\n", pca.explained_variance_ratio_)

# To recover the original data from the PCA-transformed data:
X_recovered = pca.inverse_transform(X_pca)
print("Recovered Data (approximation):\n", X_recovered)
plt.plot(X_recovered)
plt.show()