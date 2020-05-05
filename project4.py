import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Crimes_-_2001_to_present.csv',error_bad_lines=False)
df.drop_duplicates(subset=["ID","Case Number"],inplace=True)
#df1=df.loc[df['Primary Type'].isin(["MOTOR VEHICLE THEFT","THEFT","BURGLARY","ROBBERY"])]
#print(df1.info())

df0=df[["X Coordinate","Y Coordinate","Arrest","Year"]]
#print(df0["District"].unique())
#print(df0.info())

#df1=df0[df0["District"]==1.]
df2=df0[df0["Year"]==2019]
null_data=df2[df2.isnull().any(axis=1)]
#print(null_data.info())
df3=df2.drop(null_data.index, axis=0)
df3=df3[df3["X Coordinate"]!=0]
df4=df3[df3["Arrest"]==True]
df5=df3[df3["Arrest"]==False]



#print(df3.describe())
"""df4.plot.scatter(x="X Coordinate", y="Y Coordinate",color='r')
df5.plot.scatter(x="X Coordinate", y="Y Coordinate",color='b')
plt.show()"""

#prediction of arrest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from CV import find_best

new_index = range(len(df3))
#print(df3.index)
df_n=df3.reindex(new_index)
#print(df_n.index)
train_set, test_set = train_test_split(df_n, test_size=0.2, random_state=42)
"""train_set["X Coordinate"] = pd.to_numeric(train_set["X Coordinate"], errors='coerce')
m = train_set["X Coordinate"].mean(skipna=True)
train_set["X Coordinate"].fillna(m)

train_set["Y Coordinate"] = pd.to_numeric(train_set["Y Coordinate"], errors='coerce')
m = train_set["Y Coordinate"].mean(skipna=True)
train_set["Y Coordinate"].fillna(m)"""

train=train_set[["X Coordinate","Y Coordinate","Arrest"]].to_numpy()
train_x=train[np.ix_(range(len(train)), [0,1])]
train_y=train[np.ix_(range(len(train)), [2])]


#array=train[~np.isnan(train).any(axis=1)]
#print(train)
#train=pd.isnull(train([np.nan, 0], dtype=float))
"""from sklearn.preprocessing import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', axis =0)
imputer = SimpleImputer.fit(train[:,0:2])
train[:,0:2] = imputer.transform(train[:,0:2])
train_x=train[np.ix_(range(len(train)), [0,1])]
train_y=train[np.ix_(range(len(train)), [2])]
print(train_x)"""



#ho=train[~np.isnan(train).any(axis=1)]
#print(ho)
#train_y=train[np.ix_(range(len(train_set)), [2])]
#print(train_x)
"""nan_array = np.isnan(train_x)
not_nan_array = ~ nan_array
array_x = train_x[not_nan_array]
train_y=train_set["Arrest"].to_numpy()
nan_array = np.isnan(train_y)
not_nan_array = ~ nan_array
array_y = train_x[not_nan_array]"""

models=[KNeighborsClassifier(n_neighbors=3),KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=7)]
find_best(models,train_x,train_y,5)




