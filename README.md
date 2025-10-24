# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import dataset and print head,info of the dataset

2.check for null values

3.Import kmeans and fit it to the dataset

4.Plot the graph using elbow method

5.Print the predicted array

6.Plot the customer segments

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: GOUTHAM K 
RegisterNumber:  212223110019
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:5])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

km = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = km.fit_predict(data.iloc[:, 3:5])
data["Cluster"] = y_pred
print(y_pred)

plt.figure(figsize=(8, 6))
colors = ['red', 'black', 'blue', 'green', 'magenta']
for i in range(5):
    cluster = data[data["Cluster"] == i]
    plt.scatter(cluster["Annual Income (k$)"], cluster["Spending Score (1-100)"], 
                c=colors[i], label=f"Cluster {i}")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()
plt.grid(True)
plt.show()

```

## Output:
## 1.DATA.HEAD():

<img width="766" height="122" alt="image" src="https://github.com/user-attachments/assets/75da1d76-cfc1-4c7f-b13c-b97a5c758859" />

## 2.DATA.INF0():

<img width="766" height="122" alt="image" src="https://github.com/user-attachments/assets/611101a6-450f-4557-ac9d-dc5fb473965d" />

## 3.DATA.ISNULL().SUM():

<img width="357" height="137" alt="image" src="https://github.com/user-attachments/assets/56b4f862-400d-4980-95a0-5b53974fbdc5" />

## 4.PLOT USING ELBOW METHOD:

<img width="980" height="530" alt="image" src="https://github.com/user-attachments/assets/99d36e5c-f4ad-402c-85e7-ef06a05bf476" />

## 5.Y_PRED ARRAY:

<img width="712" height="126" alt="image" src="https://github.com/user-attachments/assets/201d66a1-ef32-4bc8-bdcd-201aa6740e28" />

## 6.CUSTOMER SEGMENT:

<img width="884" height="609" alt="image" src="https://github.com/user-attachments/assets/39bac0f4-fb7e-4f81-8522-cf162a5cfb26" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
