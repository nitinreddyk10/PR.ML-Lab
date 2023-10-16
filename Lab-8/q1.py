import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from collections import Counter

rd=pd.read_csv("gender.csv")
df=pd.DataFrame(rd)
df.head()

df=df.drop('Unnamed: 0',axis=1)
df.head()

Y=df['Unnamed: 1'].values 
df1=df.drop("Unnamed: 1",axis=1)
cov_matrix=df1.cov()
cov_matrix=np.array(cov_matrix)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

def trans(eigenvectors,d,df1):
    eigenvectors=eigenvectors[:d]
    df1=np.array(df1)
    New_data=np.dot(eigenvectors,df1.T)
    New_data=New_data.T
    return New_data

normalized_eigenvalues = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(normalized_eigenvalues)
cummualtive_sum=np.array(cumulative_variance)
componnets=np.array([i+1 for i in range(len(eigenvalues))])
x=np.min(np.where(cummualtive_sum>0.95))
plt.plot(componnets,cummualtive_sum)
plt.plot(x,0.95,'o')
plt.show()

df1=trans(eigenvectors,x,df1)
df1=pd.DataFrame(df1)
df1

df2=df1[:10]
df_test=df2.add(df1[790:])
df_train=np.array(df1[10:790])
y2=list(Y[:10])
y3=list(Y[790:])
y_test=y2+y3
y_train=Y[10:790]

def knn_classify(features, labels, new_data_point, k):
    distances = [np.sqrt(np.sum((new_data_point - data_point) ** 2)) for data_point in features]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [labels[i] for i in k_indices]
    label_counts = Counter(k_nearest_labels)
    predicted_label = label_counts.most_common(1)[0][0]
    return predicted_label

def mea_acc(pred,test):
    sum=0
    leng=len(pred)
    for i,j in zip(pred,test):
        if(i==j):
            sum+=1
    return (sum/leng)*100
            

predictions=[]
k=int(input('Enter the k nearest neihbors\n'))
for i in df_test:
    predict=knn_classify(df_train,y_train,i,k)
    predictions.append(predict)
acc=mea_acc(predictions,y_test)
print("Accuracy:",acc)

