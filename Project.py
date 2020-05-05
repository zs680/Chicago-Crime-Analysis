import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Crimes_-_2001_to_present.csv',error_bad_lines=False)
df=df.loc[:2000]
df.drop_duplicates(subset=["ID","Case Number"],inplace=True)


df0=df[["Location Description","Community Area"]]

null_data=df0[df0.isnull().any(axis=1)]
#print(null_data.info())
df1=df0.drop(null_data.index, axis=0)
#print(df["Location Description"].unique())
df1["Outdoor Crime"]=df1["Location Description"].isin(['SIDEWALK','STREET','PUBLIC','ALLEY','CTA STATION',
        'ATM (AUTOMATIC TELLER MACHINE)','CONSTRUCTION SITE','GAS STATION','CTA PLATFORM',
        'LAKEFRONT/WATERFRONT/RIVERBANK','CTA BUS STOP','CTA SUBWAY STATION'])
df_outdoor=df1.loc[df1["Outdoor Crime"]==True]
#df1=df_outdoor.dropna(subset=["Community Area"])


df2=df_outdoor[["Outdoor Crime","Community Area"]]



for i in df2["Community Area"].values:
    A=len(df2[df2["Community Area"]==i])
    df2.loc[df2["Community Area"]==i,"# of out door crime"]=A

print(df2)


#Create values and labels for bar chart
values =df2["# of out door crime"]
inds   =df2["Community Area"]


#Plot a bar chart
plt.figure(1, figsize=(6,4))
plt.bar(inds, values, align='center')
#plt.grid(True) #Turn the grid on
plt.ylabel("# of out door crime")
plt.xlabel("Community Area")
plt.title("number of outdoor crime per community Area")



plt.savefig("bar_chart.pdf")

plt.show()












