#!/usr/bin/env python
# coding: utf-8

# # Importimi i librarive

# In[1]:


import re
import sys
import time
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams


# # Leximi i datasetit

# In[2]:


#Importimi i datasetit
data_file = pd.read_csv("googleplaystore.csv")


# In[3]:


#Shohim madhësinë e datasetit
print(data_file.shape)


# In[4]:


#Informata mbi atributet e datasetit
data_file.info()


# # Pastrimi i të dhënave

# In[5]:


#Kontrollojmë për atributet që kanë vlera null
plt.figure(figsize=(7, 5))
sns.heatmap(data_file.isnull(), cmap='plasma')
data_file.isnull().any()


# In[6]:


#Kontrollojmë totalin e rekordeve me vlerë null për secilin atribut
data_file.isnull().sum()


# In[7]:


#Radhitim atributet sipas numrit të vlerave null dhe shohim se atributi Rating ka më së shumti vlera të tilla
total = data_file.isnull().sum().sort_values(ascending=False)
percent = (data_file.isnull().sum()/data_file.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()


# In[8]:


#Mbushim vlerat null të rekordeve përkatëse të atributit Rating me medianën e vlerave
data_file['Rating'] = data_file['Rating'].fillna(data_file['Rating'].median())
data_file.tail()


# In[9]:


#Kontrollojmë totalin e rekordeve me vlerë null për secilin atribut
data_file.isnull().sum()


# In[10]:


#Meqë numri i vlerave null për atributet tjera është shumë i vogël atëherë vedosim ti fshijmë ato 
data_file.dropna(how ='any', inplace = True)


# In[11]:


#Kontrollojmë totalin e rekordeve me vlerë null për secilin atribut
data_file.isnull().sum()


# In[12]:


#Shohim madhësinë e datasetit pas pastrimit të vlerave null
data_file.shape


# In[13]:


#Shohim të dhëna të rëndësishme për atribute numerike e të tillë e kemi vetëm atributin Rating
data_file.describe()


# In[14]:


#Duhet te behen disa ndryshime te atributeve me qëllim të pastrimit
data_file['Installs'] = data_file['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
data_file['Installs'] = data_file['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
data_file['Installs'] = data_file['Installs'].apply(lambda x: int(x))

data_file['Reviews'] = data_file['Reviews'].apply(lambda x: int(x))


data_file['Size'] = data_file['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
data_file['Size'] = data_file['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)
data_file['Size'] = data_file['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)

data_file['Installs'] = data_file['Installs'].apply(lambda x: float(x))

data_file['Price'] = data_file['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
data_file['Price'] = data_file['Price'].apply(lambda x: float(x))


# In[15]:


#Shikojmë shpërndarjen e vlerave të atributit Rating
rcParams['figure.figsize'] = 11.7,8.27
g = sns.kdeplot(data_file.Rating, color="Red", shade = True)
g.set_xlabel("Rating")
g.set_ylabel("Frekuenca")
plt.title('Shpërndarja e Rating-ut',size = 20)


# In[16]:


#Kontrollojmë nëse ka anomali në vlerat për atributin Rating
(data_file['Rating']>5.0).sum()
i = data_file[data_file['Rating'] > 5.0].index
data_file.loc[i]


# In[17]:


#Atributet me vlerë për Rating më të madhe se 5 fshihen meqë një vlerë e tillë nuk duhet të ekzistojë dhe paraqet anomali
data_file = data_file.drop(i)


# In[18]:


#Shohim formatin e të dhënave pas pastrimit
data_file.head()


# In[19]:


#Shohim shpërndarjen e Rating, Reviews dhe Price. Shihet se vlerat për Reviews anojnë kah zero, njësoj edhe për Price meqë
#shumica e aplikacioneve janë pa pagesë
f,(ax1,ax2,ax3) = plt.subplots(ncols=3,sharey=False)
sns.distplot(data_file['Rating'],hist=True,color='r',ax=ax1)
sns.distplot(data_file['Reviews'],hist=True,color='g',ax=ax2)
sns.distplot(data_file['Price'],hist=True,color='b',ax=ax3)
f.set_size_inches(15, 5)


# In[20]:


#Shikojmë numrin e aplikacioneve ekzistuese për kategori
f,ax1 = plt.subplots(ncols=1)
sns.countplot("Category", data=data_file,ax=ax1,order=data_file['Category'].value_counts().index)
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
f.set_size_inches(25,10)
ax1.set_title("Numri i aplikacioneve për kategori",size = 20)


# In[21]:


#Shikojmë vlerat e atributit Tybe dhe shohim se shumë pak aplikacione janë me pagesë

labels =data_file['Type'].value_counts(sort = True).index
sizes = data_file['Type'].value_counts(sort = True)


colors = ["palegreen","red"]
explode = (0.1,0)  # explode 1st slice
 
rcParams['figure.figsize'] = 8,8

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=270,)

plt.title('Percent of Free App in PlayStore',size = 20)
plt.show()


# In[23]:


#Shikojmë numrin e instalimeve për kategori dhe shohim se më së shumti përdoren aplikacionet për komunikim
fig = plt.figure(figsize=(12,7))
title=plt.title('Instalimet për kategori',size=20)
bar=sns.barplot(y=data_file['Category'],x=data_file['Installs'], color='lightgreen')
plt.show(bar)


# In[24]:


#Shohim varshmërinë e atributeve me njëra tjetrën
x = data_file['Rating'].dropna()
y = data_file['Size'].dropna()
z = data_file['Installs'][data_file.Installs!=0].dropna()
p = data_file['Reviews'][data_file.Reviews!=0].dropna()
t = data_file['Type'].dropna()
price = data_file['Price']

p = sns.pairplot(pd.DataFrame(list(zip(x, y, np.log(z), np.log10(p), t, price)), 
                        columns=['Rating','Size', 'Installs', 'Reviews', 'Type', 'Price']), hue='Type', palette="Set2")


# In[25]:


#Shohim konvergjencën e vlerave për atributin Rating
data_file.hist(column='Rating')
plt.ylim(0,10841)
plt.title("Shpërndarja e Rating")
plt.xlabel("Vlera e Rating")
plt.ylabel("Nr. i aplikacioneve")


# In[26]:


#Enkodimi i atributit App
le = preprocessing.LabelEncoder()
data_file['App'] = le.fit_transform(data_file['App'])
data_file


# In[27]:


# Enkodimi i atributit Category
category_list = data_file['Category'].unique().tolist() 
category_list = ['cat_' + word for word in category_list]
data_file = pd.concat([data_file, pd.get_dummies(data_file['Category'], prefix='cat')], axis=1)
data_file


# In[28]:


# Enkodimi i atributit Genre
le = preprocessing.LabelEncoder()
data_file['Genres'] = le.fit_transform(data_file['Genres'])
data_file


# In[29]:


# Enkodimi i atributit Content Rating 
le = preprocessing.LabelEncoder()
data_file['Content Rating'] = le.fit_transform(data_file['Content Rating'])
data_file.head()


# In[30]:


# Enkodimi i atributit Type
data_file['Type'] = pd.get_dummies(data_file['Type'])
data_file.head()


# In[31]:


# Enkodimi i atributit Last Updated 
data_file['Last Updated'] = data_file['Last Updated'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%B %d, %Y').timetuple()))
data_file['Last Updated'].head()


# In[32]:


# Enkodimi i atributit Size
data_file[data_file['Size'] == 'Varies with device'] = 0
data_file['Size'] = data_file['Size'].astype(float)


# In[33]:


#Mbushja e vlerave null të atributit Size me vlerën mesatare të të gjitha vlerave ekzistuese
data_file['Size'] = data_file['Size'].fillna(data_file['Size'].mean())


# In[34]:


#Mbushja e vlerave null, largimi i simboleve speciale dhe validimi i vlerave të atributit Current Ver
replaces = [u'\u00AE', u'\u2013', u'\u00C3', u'\u00E3', u'\u00B3', '[', ']', "'"]
for i in replaces:
	data_file['Current Ver'] = data_file['Current Ver'].astype(str).apply(lambda x : x.replace(i, ''))

regex = [r'[-+|/:/;(_)@]', r'\s+', r'[A-Za-z]+']
for j in regex:
	data_file['Current Ver'] = data_file['Current Ver'].astype(str).apply(lambda x : re.sub(j, '0', x))

data_file['Current Ver'] = data_file['Current Ver'].astype(str).apply(lambda x : x.replace('.', ',',1).replace('.', '')
                                                                      .replace(',', '.',1)).astype(float)
data_file['Current Ver'] = data_file['Current Ver'].fillna(data_file['Current Ver'].median())


# In[35]:


# Ndarja e të dhënave në të dhëna trajnuese dhe testuese
features = ['App', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver']
features.extend(category_list)
X = data_file[features]
y = data_file['Rating']


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)


# In[37]:


#Importimi i algoritmit SVM
from sklearn import svm


# In[38]:


#Trajnimi i algoritmit
model = svm.SVR(kernel = 'rbf')
model.fit(X_train, y_train)


# In[39]:


#Saktësia e algoritmit
model.score(X_test, y_test)


# In[40]:


#Vlerat e predikuara nga algoritmi SVM
Y_out=model.predict(X_test)
print(Y_out)


# In[41]:


#Krahasimi i vlerës së predikuar nga algoritmi SVM për atributin Rating me vlerën aktuale të tij nga të dhënat testuese
for x,y in zip(Y_out,y_test):
    print(f"Predicted:{round(x,1)}     Actual:{y}")


# In[42]:


#Mean absolute percentage error
mape_sum = 0
Y_out=model.predict(X_test)
for x,y in zip(y_test,Y_out):
        mape_sum += (abs(x - y)/y)
        
mape = mape_sum/len(y_test) *100

print(f"MAPE: {round(mape,2)}%")


# In[43]:


#Mean percentage error
mpe_sum = 0
Y_out=model.predict(X_test)
for x, y in zip(y_test, Y_out):
    mpe_sum += ((x - y)/x)
    
mpe = mpe_sum/len(y_test) *100

print(f"MPE: {round(mpe,2)}")


# In[44]:


#Shkalla e gabimit
accurracy=model.score(X_test,y_test)
error=1-accurracy
error_percentage=error*100
print(f"Shkalla e gabimit në raport me saktësinë e algoritmit është {round(error_percentage,2)}%")


# In[45]:


#Shohim konvergjencën e vlerave të predikuara për atributin Rating
plt.hist(Y_out,5)
plt.ylim(0,2708)
plt.xlim(0,5)
plt.title("Shpërndarja e vlerave të predikuara")
plt.xlabel("Vlera e rating")
plt.ylabel("Nr. i aplikacioneve")

