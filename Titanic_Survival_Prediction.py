#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


pip install xgboost


# In[4]:


pip install lightgbm


# In[6]:


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# In[4]:


train.head()


# In[5]:


train.describe()


# In[6]:


train.info()


# In[7]:


sns.countplot(x=train["Survived"])


# In[8]:


train.info()


# In[9]:


sns.countplot(x=train["Pclass"])


# In[10]:


sns.countplot(x=train["Sex"])


# In[11]:


sns.histplot(x=train["Age"])


# In[27]:


df=pd.concat([train, test],axis=0)
df=df.reset_index(drop=True)


# In[28]:


df.info()


# In[29]:


df.shape


# In[30]:


df.isnull().sum()


# In[31]:


df=df.drop(columns=["Cabin"], axis=0)


# In[32]:


df.isnull().sum()


# In[33]:


df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Fare"]=df["Fare"].fillna(df["Fare"].mean())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])


# In[34]:


df.isnull().sum()


# In[35]:


df["Fare"]=np.log(df["Fare"]+1)


# In[36]:


sns.displot(df["Fare"])


# In[39]:


corr=df.corr()
plt.figure(figsize=(15,9))
sns.heatmap(corr, annot=True, cmap="PiYG")


# In[40]:


sns.barplot(data=train, x="Pclass", y="Fare", hue="Survived")


# In[41]:


df=df.drop(columns=["Name", "Ticket"], axis=1)
df


# In[43]:


from sklearn.preprocessing import LabelEncoder
cols=["Sex", "Embarked"]
le=LabelEncoder()

for col in cols:
    df[col]=le.fit_transform(df[col])
df


# In[45]:


train_len = len(train)
train=df.iloc[:train_len, :]
test=df.iloc[train_len:, :]


# In[46]:


test.head()


# In[48]:


X=train.drop(columns=["PassengerId", "Survived"], axis=1)
y=train["Survived"]


# In[49]:


X.head()


# In[60]:


from sklearn.model_selection import train_test_split, cross_val_score
def classify(model):
    x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print("Accuracy:", model.score(x_test, y_test))
    
    score=cross_val_score(model, X, y, cv=5)
    print("CV score:", np.mean(score))


# In[62]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
classify(model)


# In[63]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model)


# In[64]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model)


# In[65]:


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
classify(model)


# In[71]:


from xgboost import XGBClassifier
model = XGBClassifier()
classify(model)


# In[77]:


from lightgbm import LGBMClassifier
model = LGBMClassifier(verbose=0)
classify(model)


# In[78]:


model=LGBMClassifier()
model.fit(X, y)


# In[79]:


test.head()


# In[82]:


X_test=test.drop(columns=["PassengerId", "Survived"], axis=1)


# In[83]:


X_test.head()


# In[84]:


pred=model.predict(X_test)
pred


# In[85]:


final=pd.read_csv('gender_submission.csv')


# In[86]:


final.head()


# In[87]:


final["Survived"]=pred


# In[88]:


final.head()


# In[89]:


final.to_csv("Final_submission.csv")


# In[ ]:




