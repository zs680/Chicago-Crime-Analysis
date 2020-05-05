import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Crimes_-_2001_to_present.csv',error_bad_lines=False)
df.drop_duplicates(subset=["ID","Case Number"],inplace=True)
df1=df[["ID","Primary Type","District", "Arrest"]]
"""print(df1.describe())
print(df1.info())"""
set_district=df1["District"].unique()
#print(set_district)
for district in set_district:
    A=df1.loc[df1["District"]==district]
    df1.loc[df1["District"]==district,"#crime_dist"]=len(A)
    df1.loc[df1["District"]==district,"#arrest_dist"]=len(A.loc[A["Arrest"]==True])
    df1.loc[df1["District"]==district,"average_arrest_dist"]=len(A.loc[A["Arrest"]==True])/len(df1.loc[df1["District"]==district])

#print(df1[["District","average_arrest_dist"]])
A=df1[["District","average_arrest_dist"]]

B=A.sort_values("District")
plt.plot(B["District"], B["average_arrest_dist"])
plt.title('average arrest', fontsize=8, color='g')
plt.xlabel('District')
plt.ylabel('Average arrest')
plt.show()
plt.savefig('Average.png')
