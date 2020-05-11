import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Crimes_-_2001_to_present.csv',error_bad_lines=False)
#df=df.loc[:1000000]
df.drop_duplicates(subset=["ID","Case Number"],inplace=True)


df0=df[["Location Description","Community Area","Date"]]

# convert dates to pandas datetime format

df0['Date'] = pd.to_datetime(df0['Date'], format='%m/%d/%Y %H:%M:%S %p')
df0 ['Year'] = df0['Date'].dt.year
df0 ['Month'] = df0['Date'].dt.month
df0 ['Day'] = df0['Date'].dt.day

null_data=df0[df0.isnull().any(axis=1)]
#print(null_data.info())
df1=df0.drop(null_data.index, axis=0)
#print(df["Location Description"].unique())
df1["Outdoor Crime"]=df1["Location Description"].isin(['SIDEWALK','STREET','PUBLIC','ALLEY','CTA STATION',
        'ATM (AUTOMATIC TELLER MACHINE)','CONSTRUCTION SITE','GAS STATION','CTA PLATFORM',
        'LAKEFRONT/WATERFRONT/RIVERBANK','CTA BUS STOP','CTA SUBWAY STATION'])



def f(Year):
    df11=df1.loc[df1["Year"]==Year]
    df_outdoor=df11.loc[df11["Outdoor Crime"]==True]
    #df1=df_outdoor.dropna(subset=["Community Area"])
    df2=df_outdoor[["Community Area"]]

    for i in df2["Community Area"].values:
        A=len(df2[df2["Community Area"]==i])
        df2.loc[df2["Community Area"]==i,"# of out door crime"]=A

    return df2


#Create values and labels for bar chart
#df2=f(2019)
#values =df2["# of out door crime"]
#Mod=values.mod()

#max=values.max()
#print(df2.loc[df2["# of out door crime"]==max])


plt.figure( figsize=(10,10))
fig, plots = plt.subplots(2)
for i in range(0,4):
    Year=2018+i
    df2=f(Year)
    df2.to_csv('Project {}.CSV'.format(Year))
    values =df2["# of out door crime"]
    inds =df2["Community Area"]
    plots[i].bar(inds, values, align='center')
    plots[i].set_title("Year {}".format(Year))
    plt.ylabel("Number of Outdoor Crimes")
    plt.xlabel("Community Area")
plt.savefig("bar_chart_Project.pdf")
plt.show()

"""Year=2016
df2=f(Year)
df2.to_csv('Project {}.CSV'.format(Year))
plt.figure(1, figsize=(10,4))
values =df2["# of out door crime"]
inds =df2["Community Area"]
plt.bar(inds, values, align='center')
#plt.grid(True) #Turn the grid on
plt.ylabel("# of out door crime")
plt.xlabel("Community Area")
plt.title("Year {}".format(Year))
plt.savefig("bar_chart_Project.pdf")
plt.show()"""










