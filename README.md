#### Tumor Detection Using RandomForest Algorithm

### AIM

    To create a Tumor detection model with random forest algorithm
### Steps

### 1.Importing libraries

 Import libraries such as pandas,numpy,matplot,seaborn

 ```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2.Read the Csv file

```
df=pd.read_csv("Tumor_detection.csv")
```

### 3.Drop the unnecessary columns

```
df.drop('id',axis=1,inplace=True)
```

### 4.Put all the columns into a list

```
l=list(df.columns)
print(l)
```

### 5.Creating start points
```
features_mean=l[1:11]
features_se=l[11:21]
features_worst=l[21:]
```

### 6.Plot the diagnosis column

 ploting the diagnosis column using countplot to check how many malignant and begin stage of tumor are in the dataset
 
 ```
 sns.countplot(x=df['diagnosis'],data=df)
```

### 7.Checking correlation

```
numeric_df = df.select_dtypes(include=['float64', 'int64']) # as we have string in diagnosis column we are selecting the dtype
corr = numeric_df.corr()
corr
```

### 8.Heatmap

```
plt.figure(figsize=(10,10))
sns.heatmap(corr)
```

### 9.Changing M to 1 and B to 0

```
df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
df.head()
```

### 10.

```
x=df.drop('diagnosis',axis=1)
y=df['diagnosis']






