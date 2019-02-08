#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:49:22 2019

@author: ohwadalab-mac
"""


# coding: utf-8


# Import package
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.style.use('seaborn-whitegrid')
from mpl_toolkits.mplot3d import axes3d


# Import dataset
df = pd.read_csv('./dairy-dry.csv',sep=',')

# Check dataset
df.head()

# Make copy of dataset to be used for clustering
df_tr = df

# Standardize
# Calculate the z score of each value in the sample, 
# relative to the sample mean and standard deviation.
clmns = ['TP','ALB','BUN','GOT', 'GGT', 'CA','IP','GLU','NH3', 'NEFA','BHB']
df_tr_std = stats.zscore(df_tr[clmns])

# Deciding how many cluster needed using Elbow Method
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_tr)
    sse[k] = kmeans.inertia_ 
    # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

# Check the Silhoutte Coefficient to confirm the number of cluster
# Generally, chooose n_cluster with highest score, but other considerations
# depending on the data and analysis can be taken

from sklearn.metrics import silhouette_samples, silhouette_score

for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(df_tr)
    label = kmeans.labels_
    sil_coeff = silhouette_score(df_tr, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

# Implement k-means to standardized data 
kmeans = KMeans(n_clusters=2, random_state=0).fit(df_tr_std)
labels = kmeans.labels_

# After cluster is formed, add new column defining their cluster
df_tr['clusters'] = labels
clmns.extend(['clusters'])

# Check result of clustering
df_tr

# If needed, save clustering result
# df_tr.to_csv('./cluster-dairy-dry.csv')

# Calculate mean of each cluster
mean = df_tr[clmns].groupby(['clusters']).mean()
mean

# If needed, save mean result
# mean.to_csv('./meanbefore.csv')

# Calculate standard deviation of each cluster
std = df_tr[clmns].groupby(['clusters']).std()
std

# If needed, save stdev result
# std.to_csv('./std1.csv')


# Plot 1: Relationship between Calcium and Phosphorus within each cluster

ax = sns.lmplot('CA', 'IP', 
          data=df_tr, 
         fit_reg=False, 
        hue="clusters",  palette="Set1",
       scatter_kws={"marker": "D", 
                   "s": 50})

plt.title('Calcium vs Phosporus (Mineral Metabolism)')
plt.xlabel('CA')
plt.ylabel('IP')


# Plot 2: 3D - plot between NEFA, GLU, and BHB within each cluster

fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111,projection='3d')

colors = ['red', 'blue', 'green']
for i in range(len(df_tr)):
    x, y, z = df_tr.iloc[i]['BHB'], df_tr.iloc[i]['NEFA'], df_tr.iloc[i]['GLU']
    ax1.scatter(x, y, z, c=colors[int(df_tr.iloc[i]['clusters'])], alpha = 0.6, linewidths=4, label = 'clusters')
    ax1.text(x, y, z, '{0}'.format(df_tr.iloc[i]['cowid'].astype(int)),va='bottom', size=10)
 
    
ax1.set_xlabel('BHB')
ax1.set_ylabel('NEFA')
ax1.set_zlabel('GLU')

plt.title('Clusters in dry period related to Energy Balance')    
plt.show()

# For view from different angle
#for angle in range(0, 360):
#    ax1.view_init(10, angle)
#    plt.draw()
#    plt.pause(.001)

# Plot 3: 3D - plot between Total Protein, ALB, and BUN within each cluster

fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111,projection='3d')

colors = ['red', 'blue', 'green']
for i in range(len(df_tr)):
    x, y, z = df_tr.iloc[i]['TP'], df_tr.iloc[i]['ALB'], df_tr.iloc[i]['BUN']
    ax1.scatter(x, y, z, c=colors[int(df_tr.iloc[i]['clusters'])], alpha = 0.6, linewidths=4)
    ax1.text(x, y, z, '{0}'.format(df_tr.iloc[i]['cowid'].astype(int)),va='bottom', size=10)
 
    
ax1.set_xlabel('TP')
ax1.set_ylabel('ALB')
ax1.set_zlabel('BUN')

plt.title('Clusters in dry period related to Protein Status')   
plt.show()

# For view from different angle
#for angle in range(0, 360):
#       ax1.view_init(10, angle)
#       plt.draw()
#       plt.pause(.001)

# Plot 4: 3D - plot between GOT, GGT, and NH3 within each cluster

fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111,projection='3d')

colors = ['red', 'blue', 'green']
for i in range(len(df_tr)):
    x, y, z = df_tr.iloc[i]['GOT'], df_tr.iloc[i]['GGT'], df_tr.iloc[i]['NH3']
    ax1.scatter(x, y, z, c=colors[int(df_tr.iloc[i]['clusters'])], alpha = 0.6, linewidths=4)
    ax1.text(x, y, z, '{0}'.format(df_tr.iloc[i]['cowid'].astype(int)),ha='right', size=10)
    
    
ax1.set_xlabel('GOT')
ax1.set_ylabel('GGT')
ax1.set_zlabel('NH3')


plt.title('Clusters in dry period related to Liver Condition')      
#plt.show()

# For view from different angle
#for angle in range(0, 360):
#       ax1.view_init(10, angle)
#       plt.draw()
#       plt.pause(.001)

'''
# Pair plot to see relationship between 2 metabolites (if needed)
# Energy Balance
energy = df_tr[['BHB', 'NEFA', 'GLU','clusters']]

sns.pairplot(energy, kind="scatter",  diag_kind="hist", hue="clusters", markers=["o", "o"], palette="Set1")
plt.show()

# Protein
pro = df_tr[['TP', 'BUN', 'ALB','clusters']]

sns.pairplot(pro, kind="scatter",  diag_kind="hist", hue="clusters", markers=["o", "o"], palette="Set1")
plt.show()

# Liver Condition
liver = df_tr[['NH3', 'GGT', 'GOT','clusters']]

sns.pairplot(liver, kind="scatter",  diag_kind="hist", hue="clusters", markers=["o", "o"], palette="Set1")
plt.show()

# Sort the dataframe by clusters
df_tr = df.sort_values(by="clusters")
'''

# Define limit of each metabolites and map with color
def color_protein(value):
    if value < 6:
        color = 'red'
    elif 6.8 < value or value > 8.5:
        color = 'green'
    else:
        color = 'black'

    return 'color: %s' % color

def color_alb(value):
    if value < 3:
        color = 'red'
    elif value > 3.7:
        color = 'green'
    else:
        color = 'black'

    return 'color: %s' % color

def color_bun(value):
    if value < 13:
        color = 'red'
    elif value > 17:
        color = 'green'
    else:
        color = 'black'

    return 'color: %s' % color

def color_nefa(value):
    if value > 700:
        color = 'red'
    elif value < 300:
        color = 'green'
    else:
        color = 'black'

    return 'color: %s' % color

def color_glu(value):
    if value < 42:
        color = 'red'
    elif value > 74:
        color = 'green'
    else:
        color = 'black'

    return 'color: %s' % color

def color_ggt(value):
    if value > 27:
        color = 'red'
    elif  value < 15 or value > 19:
        color = 'green'
    else:
        color = 'black'

    return 'color: %s' % color


def color_ca(value):
    if value < 8:
        color = 'red'
    elif value > 11:
        color = 'green'
    else:
        color = 'black'

    return 'color: %s' % color


def color_ip(value):
    if value < 3.5:
        color = 'red'
    elif value > 8:
        color = 'green'
    else:
        color = 'black'

    return 'color: %s' % color

def color_bhb(value):
    if value > 960:
        color = 'red'
    elif value < 120:
        color = 'green'
    else:
        color = 'black'

    return 'color: %s' % color

def color_got(value):
    if value > 100:
        color = 'red'
    else:
        color = 'black'

    return 'color: %s' % color


new_df = df_tr.drop(columns=['CHO'])    


# Map individual data based on metabolites score limit
# Output is saved in your directory (xls file)

#new_df.style.applymap(color_protein, subset=['TP']).    applymap(color_alb, subset=['ALB']).    applymap(color_nefa, subset=['NEFA']).    applymap(color_ggt, subset=['GGT']).    applymap(color_ca, subset=['CA']).    applymap(color_ip, subset=['IP']).    applymap(color_glu, subset=['GLU']).    applymap(color_bhb, subset=['BHB']).    applymap(color_got, subset=['GOT']).    applymap(color_bun, subset=['BUN']). to_excel('dairydry.xlsx', engine='openpyxl')


# Print out summary 
def main():
    cluster0= df_tr["global"][(df_tr["clusters"] == 0)].tolist()
    cluster1= df_tr["global"][(df_tr["clusters"] == 1)].tolist()
    cluster2= df_tr["global"][(df_tr["clusters"] == 2)].tolist()
    nc0 = len(df_tr[(df_tr['clusters']==0)])
    nc1 = len(df_tr[(df_tr['clusters']==1)])
    nc2 = len(df_tr[(df_tr['clusters']==2)])
    
    cnteb = len(df_tr[(df_tr['clusters'] == 0) & ((df_tr["NEFA"] > 460)| (df_tr["GLU"] < 51) | (df_tr["BHB"] > 403))])
    cntpro = len(df_tr[(df_tr['clusters'] == 0) & ((df_tr["TP"] < 6.8)| (df_tr["BUN"] < 13) | (df_tr["ALB"] < 3.3))])
    cntgot = len(df_tr[(df_tr['clusters'] == 0) & ((df_tr["GOT"] > 100)| (df_tr["GGT"] > 27))])
    cntca = len(df_tr[(df_tr['clusters'] == 0) & ((df_tr["CA"] < 8) | (df_tr["IP"] < 3.5))])
    
    cnteb1 = len(df_tr[(df_tr['clusters'] == 0) & ((df_tr["NEFA"] > 460)| (df_tr["GLU"] < 51) | (df_tr["BHB"] > 403))])
    cntpro1 = len(df_tr[(df_tr['clusters'] == 0) & ((df_tr["TP"] < 6.8)| (df_tr["BUN"] < 13) | (df_tr["ALB"] < 3.3))])
    cntgot1 = len(df_tr[(df_tr['clusters'] == 0) & ((df_tr["GOT"] > 100)| (df_tr["GGT"] > 27))])
    cntca1 = len(df_tr[(df_tr['clusters'] == 1) & ((df_tr["CA"] < 8) | (df_tr["IP"] < 3.5))])
    
    cnteb2 = len(df_tr[(df_tr['clusters'] == 0) & ((df_tr["NEFA"] > 460)| (df_tr["GLU"] < 51) | (df_tr["BHB"] > 403))])
    cntpro2 = len(df_tr[(df_tr['clusters'] == 0) & ((df_tr["TP"] < 6.8)| (df_tr["BUN"] < 13) | (df_tr["ALB"] < 3.3))])
    cntgot2 = len(df_tr[(df_tr['clusters'] == 2) & ((df_tr["GOT"] > 100)| (df_tr["GGT"] > 27))])
    cntca2 = len(df_tr[(df_tr['clusters'] == 2) & ((df_tr["CA"] < 8) | (df_tr["IP"] < 3.5))])
    
    
    print("Cows in dry period (before calving time):")
    print("---")
    print("Cluster 0 have", nc0, "cows with ID:", cluster0)
    if cntpro > 0.7* nc0:
        print ("Food and diet should be monitored because protein is out of range.")
    else:
        print ("Protein status is in good condition.")    
    if cntgot > 0.7 * nc0:
        print ("Liver condition should be monitored in next 4-6 weeks because liver value is out of range, glucose infusion can be considered.")
    else:
        print ("Liver condition is in good condition.")    
    if cnteb > 0.7*nc0:
        print ("Some cow has negative energy balance, high-energy feed intake may be needed.")
    else :
        print("Energy balance is in good condition.")
    if cntca > 0.7*nc0:
        print ("Cow has high calcium diet demand.")
    else :
        print("Calcium and phosphorus is in good condition.")
    print("---")
    print("Cluster 1 have", nc1, "cows with ID:", cluster1)
    if cntpro1 > 0.7* nc1:
        print ("Food and diet should be monitored because protein is out of range.")
    else:
        print ("Protein status is in good condition.")    
    if cntgot1 > 0.7 * nc1:
        print ("Liver condition should be monitored in next 4-6 weeks because liver value is out of range, glucose infusion can be considered.")
    else:
        print ("Liver condition is in good condition.")    
    if cnteb1 > 0.7*nc1:
        print ("Some cow has negative energy balance, high-energy feed intake may be needed.")
    else :
        print("Energy balance is in good condition.")
    if cntca1 > 0.7*nc1:
        print ("Cow has high calcium diet demand.")
    else :
        print("Calcium and phosphorus is in good condition.")
    print("---")
    

if __name__ == "__main__":
    main()
