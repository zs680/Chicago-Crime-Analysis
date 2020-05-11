import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Crimes_-_2001_to_present.csv',error_bad_lines=False)
#df=df.loc[:1000000]
df.drop_duplicates(subset=["ID","Case Number"],inplace=True)
df0=df[["ID","Primary Type","District", "Arrest","X Coordinate","Y Coordinate", "Year"]]
null_data=df0[df0.isnull().any(axis=1)]
#print(null_data.info())
df00=df0.drop(null_data.index, axis=0)

df_S=df00.loc[df00["Year"]==2019]
df_S=df_S.loc[df_S["X Coordinate"]!=0]
plt.scatter(df_S["X Coordinate"], df_S["Y Coordinate"],c=df_S["Arrest"], s=.1)

plt.show()
plt.savefig('scatter.pdf')

def g (year):
    df1=df00.loc[df00["Year"]==year]
    df1=df1.loc[df1["X Coordinate"]!=0]
    set_district=df1["District"].unique()
    #print(set_district)
    for district in set_district:
        A=df1.loc[df1["District"]==district]
        df1.loc[df1["District"]==district,"#crime_dist"]=len(A)
        df1.loc[df1["District"]==district,"#arrest_dist"]=len(A.loc[A["Arrest"]==True])
        df1.loc[df1["District"]==district,"average_arrest_dist"]=len(A.loc[A["Arrest"]==True])/len(df1.loc[df1["District"]==district])
    return df1
    #print(df1[["District","average_arrest_dist"]])
plt.figure()
fig, ax = plt.subplots(figsize=(10,10))
for i in range(0,4):
    year=2016+i
    df1=g(year)
    A=df1[["District","average_arrest_dist"]]
    B=A.sort_values("District")
    ax.plot(B["District"], B["average_arrest_dist"],label=year)
plt.title('average arrest per district', fontsize=8, color='g')
plt.xlabel('District')
plt.ylabel('Average arrest')
plt.legend()
plt.savefig('Average.pdf')
plt.show()







from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
train_x=df1[["X Coordinate","Y Coordinate"]]
df1.loc[df1["Arrest"]==True,"Arrest_new"]=1
df1.loc[df1["Arrest"]==False,"Arrest_new"]=0
train_y=df1["Arrest_new"]
df2=df00.loc[df00["Year"]==2020]
test_x=df2[["X Coordinate","Y Coordinate"]]
df2.loc[df2["Arrest"]==True,"Arrest_new"]=1
df2.loc[df2["Arrest"]==False,"Arrest_new"]=0
test_y=df2["Arrest_new"]
model.fit(train_x, train_y)
print(model.predict(test_x))
