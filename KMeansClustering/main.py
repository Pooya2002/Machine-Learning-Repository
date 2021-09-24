import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

database = pd.read_csv('Housing_Data.csv')
whole_data = database.iloc[:, :].values
kmeans = KMeans(n_clusters=3).fit(whole_data)

Y_Pred = kmeans.predict(whole_data)

fig, axes = plt.subplots(1, 5, figsize=(10, 5))
index = 0
for axis in axes:
    axis.scatter(whole_data[Y_Pred == 0, index], whole_data[Y_Pred == 0, (index + 1) % 5], c='b', label='Cluster 0')
    axis.scatter(whole_data[Y_Pred == 1, index], whole_data[Y_Pred == 1, (index + 1) % 5], c='g', label='Cluster 1')
    axis.scatter(whole_data[Y_Pred == 2, index], whole_data[Y_Pred == 2, (index + 1) % 5], c='r', label='Cluster 2')
    axis.scatter(kmeans.cluster_centers_[0, index], kmeans.cluster_centers_[0, (index + 1) % 5], c='b', marker='^')
    axis.scatter(kmeans.cluster_centers_[1, index], kmeans.cluster_centers_[1, (index + 1) % 5], c='g', marker='^')
    axis.scatter(kmeans.cluster_centers_[2, index], kmeans.cluster_centers_[2, (index + 1) % 5], c='r', marker='^')
    axis.set_xlabel(str(index) + " feature")
    axis.set_ylabel(str(index + 1) + " feature")
    axis.legend()
    index = index + 1
plt.show()
