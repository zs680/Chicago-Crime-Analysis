import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#df1=df.loc[df['Primary Type'].isin(["MOTOR VEHICLE THEFT","THEFT","BURGLARY","ROBBERY"])]
#print(df1.info())


df = pd.read_csv('Crimes_-_2001_to_present.csv',error_bad_lines=False)
#df=df.loc[:1000000]
df.drop_duplicates(subset=["ID","Case Number"],inplace=True)

df0=df[["Location Description","Community Area","Year","X Coordinate","Y Coordinate","Arrest",'Primary Type']]

null_data=df0[df0.isnull().any(axis=1)]
#print(null_data.info())
df1=df0.drop(null_data.index, axis=0)

df1["Outdoor Crime"]=df1["Location Description"].isin(['SIDEWALK','STREET','PUBLIC','ALLEY','CTA STATION',
        'ATM (AUTOMATIC TELLER MACHINE)','CONSTRUCTION SITE','GAS STATION','CTA PLATFORM',
        'LAKEFRONT/WATERFRONT/RIVERBANK','CTA BUS STOP','CTA SUBWAY STATION'])

df11=df1.loc[df1["Year"]==2019]
df_outdoor=df11.loc[df11["Outdoor Crime"]==True]
df_outdoor=df_outdoor.loc[df_outdoor['Primary Type'].isin(["MOTOR VEHICLE THEFT","THEFT","BURGLARY","ROBBERY"])]
df_outdoor=df_outdoor.loc[df_outdoor["X Coordinate"]!=0]


from sklearn.cluster import KMeans
#for i in range(77):
i=30
df_c=df_outdoor.loc[df_outdoor["Community Area"]==i]
X=df_c[["X Coordinate","Y Coordinate","Arrest"]]
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
X["Label"]=kmeans.labels_
#X["Label"]=kmeans.labels_
plt.scatter(X["X Coordinate"], X["Y Coordinate"],c=X["Label"])
plt.title("Scatter plot for location distribution of community area 30 with 10 clusters")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.savefig("Cluster.pdf")
plt.show()
#kmeans.predict([[0, 0], [12, 3]])










"""plt.scatter(X["X Coordinate"], X["Y Coordinate"],c=X["Arrest"])
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.savefig("Arrest.pdf")
plt.show()

kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
X["Label"]=kmeans.labels_
plt.scatter(X["X Coordinate"], X["Y Coordinate"],c=X["Label"])
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.savefig("Cluster.pdf")
plt.show()
"""

print(kmeans.cluster_centers_)



#print(neigh.predict_proba([[0.9]]))









#prediction of arrest
"""from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from CV import find_best

new_index = range(len(df3))
#print(df3.index)
df_n=df3.reindex(new_index)
#print(df_n.index)
train_set, test_set = train_test_split(df_n, test_size=0.2, random_state=42)


train=train_set[["X Coordinate","Y Coordinate","Arrest"]].to_numpy()
train_x=train[np.ix_(range(len(train)), [0,1])]
train_y=train[np.ix_(range(len(train)), [2])]

models=[KNeighborsClassifier(n_neighbors=3),KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=7)]
find_best(models,train_x,train_y,5)"""




