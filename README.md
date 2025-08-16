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

### 10.Spliting X and Y from the dataset

here we are going to predict the Tumor using its diagnosis column 
so,we are droping that column from x and storing it in the y.(i.e) x contains input values and y contain output values


```
x=df.drop('diagnosis',axis=1)
y=df['diagnosis']
```

### 11. Train the dataframe

import  standards scaler and train_test_split from sklearn and fit train and test in the variable

```
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test= ss.fit_transform(x_test)
```

### 12. Apply Randomforest Classifier
importing randomforest and accuracyscore. accuracy_score is nothing but the  proportion of correct predictions.

```
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
```
### 13.Prediction Using Random forest classifier

```
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
print(accuracy_score(y_test,y_pred))
```

### 14.Conclusion
       Now we can predict the tumor whether it is malignant or begin






