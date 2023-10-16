import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

rd=pd.read_csv("face.csv")
df=pd.DataFrame(rd)
df.head()

uni=df['target'].values
print(np.unique(uni))

print(df.shape)

Y=df['target'].values 
df=df.drop('target',axis=1)
df

cov_matrix=df.cov()
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
# x=np.min(np.where(cummualtive_sum>0.95))
plt.plot(componnets,cummualtive_sum)
plt.show()

x=np.min(np.where(cummualtive_sum>0.95))
df1=trans(eigenvectors,x,df)
df1=pd.DataFrame(df1)
df1

num_datasets = 40
dataset_size = len(df) // num_datasets

smaller_datasets = []
for i in range(num_datasets):
    start_idx = i * dataset_size
    end_idx = (i + 1) * dataset_size
    smaller_df = df1.iloc[start_idx:end_idx]
    smaller_datasets.append(smaller_df)

train_datasets = []
test_datasets = []

for dataset in smaller_datasets:
    num_rows = len(dataset)
    num_test_rows = int(num_rows * 0.2) 
    test_indices = np.random.choice(dataset.index, num_test_rows, replace=False)
    test_set = dataset.loc[test_indices]
    train_set = dataset.drop(test_indices)
    
    train_datasets.append(train_set)
    test_datasets.append(test_set)

df_train = pd.concat(train_datasets)
df_test=pd.concat(test_datasets)

y_train=[]
y_test=[]
for i in range(40):
    for j in range(8):
        y_train.append(i)
    for j in range(2):
        y_test.append(i)
y_train=np.array(y_train)
y_test=np.array(y_test)

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
