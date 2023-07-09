from ast import increment_lineno
from functools import reduce
from hashlib import new
from random import random
from re import S
import numpy as np
import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
from sympy import re



df=pd.read_csv("../Raw CSV/Item_score_perCustomer US (no skintype score no 2010).csv")
df.reset_index
df.columns=['CustomerId','Acne_prone_skin','Breakouts','Blackheads_clogged_pores','Enlarged_pores_Oil_control','Age_prevention_Youth_Preservation','Anti_aging','Firmness','Wrinkles','Brown_spots_uneven_skin_tone','Dull_skin','Keratosis_pilaris_bumpy_skin','Redness','Senstive_skin','Very_dry_skin','Rosacea_prone_skin','Dehydrated_skin','Anti_pollution']
df.dropna(axis=0,inplace=True)



new_df=df.drop('CustomerId',axis=1).reset_index(drop=True)


print("customers dataset has {} samples with {} features each.".format(*new_df.shape))

display(new_df.corr())

sns.set(font_scale =0.8)
p1=sns.heatmap(new_df.corr())
p1.set_title("Heatmap")
#plt.show()

new_df['avg_Acne_skin+breakouts']=(new_df['Acne_prone_skin']+new_df['Breakouts'])/2
new_df1=new_df.drop(['Acne_prone_skin','Breakouts'],axis=1).reset_index(drop=True)

#########################################def pca_results ##########################################

def pca_results(good_data, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = list(good_data.keys()))
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (14,8))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar');
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)


    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)


########################### preprocessing & import PCA (Principle Component Analysis) ###############################
from sklearn.decomposition import PCA




################################### outliers ##############################################

outliers= []
for feature in new_df1.keys():
    Q1=np.percentile(new_df1[feature],25)

    Q3=np.percentile(new_df1[feature],75)

    step=1.5*(Q3-Q1)

    print("Data points considered outliers for the feature '{}':".format(feature))
    display(new_df1[~((new_df1[feature] >= Q1-step)&(new_df1[feature] <= Q3 + step))])
    outliers.append(new_df1[~((new_df1[feature] >= Q1-step)&(new_df1[feature] <= Q3 + step))].index.tolist())

flat_list=sum(outliers,[])
flat_list0=pd.Series(flat_list)[pd.Series(flat_list).duplicated()].values

flat_list2=np.array(flat_list0)
flat_list3=np.unique(flat_list2)

duplicated_outliers=flat_list3
df_scaled_good_data=new_df1.drop(new_df1.index[duplicated_outliers])

print(df_scaled_good_data.shape)


######################################## Variance Explained Graph ##########################################
pca=PCA(n_components=6, random_state=2)
pca.fit(df_scaled_good_data)
pca_results=pca_results(df_scaled_good_data,pca)
#plt.show()



###################################  Def Biplot  ###############################################
def biplot(good_data, reduced_data, pca):
    fig, ax = plt.subplots(figsize = (14,8))
    # scatterplot of the reduced data    
    ax.scatter(x=reduced_data.loc[:, 'Dimension 1'], y=reduced_data.loc[:, 'Dimension 2'], 
        facecolors='b', edgecolors='b', s=0.25, alpha=0.5)
    
    feature_vectors = pca.components_.T

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 5.0, 6.0,

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], 
                  head_width=0.05, head_length=0.05, linewidth=0.5, color='red')
        ax.text(v[0]*text_pos, v[1]*text_pos, good_data.columns[i], color='red', 
                 ha='center', va='center', fontsize=8.0)

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC plane with original Scores projections.", fontsize=16);
    return ax

###################################2-D graph,Variance Explained###############################################
pca=PCA(n_components=2,random_state=3)
pca.fit(df_scaled_good_data)
reduced_data=pca.transform(df_scaled_good_data)


reduced_data=pd.DataFrame(reduced_data,columns=['Dimension 1','Dimension 2'])
biplot(df_scaled_good_data,reduced_data,pca)

################################## GaussianMixture Clustering #########################################

from sklearn.mixture import GaussianMixture
clusterer=GaussianMixture(n_components=5,random_state=4)
clusterer.fit(reduced_data)

preds=clusterer.predict(reduced_data)

weights=clusterer.weights_
centers=clusterer.means_

print(weights)
############################################ Silhouette Score ####################################################
"""

n_clusters = [3,4,5,6,7,8,9,10]

from sklearn.metrics import silhouette_score

for n in n_clusters:
    
    # TODO: Apply clustering algorithm of choice to the reduced data 
    clusterer = GaussianMixture(n_components=n).fit(reduced_data)

    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # TODO: Find the cluster centers
    centers = clusterer.means_

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data,preds)
    
    print ("The silhouette_score for {} clusters is {}".format(n,score))


"""
################################### Elbow Rule Test #####################################################
"""
from scipy.spatial.distance import cdist

distortions = [] 
K = range(1, 11) 
for k in K: 
    GMMtest = GaussianMixture(n_components=k).fit(reduced_data) 
    GMMtest.fit(reduced_data) 
    distortions.append(sum(np.min(cdist(reduced_data, GMMtest.means_, 'euclidean'), axis=1)) / reduced_data.shape[0]) 

# Plot the elbow 
plt.plot(K, distortions, 'bx-') 
plt.xlabel('k') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method showing the optimal k')
plt.show()

"""
################################# Def cluster_results #######################################################

import matplotlib.cm as cm

def cluster_results(reduced_data, preds, centers):
    '''
    Visualizes the PCA-reduced cluster data in two dimensions
    Adds cues for cluster centers and student-selected sample data
    '''

    predictions = pd.DataFrame(preds, columns = ['Cluster'])
    plot_data = pd.concat([predictions, reduced_data], axis = 1)

    # Generate the cluster plot
    fig, ax = plt.subplots(figsize = (14,8))

    # Color map
    cmap = cm.get_cmap('gist_rainbow')

    # Color the points based on assigned cluster
    for i, cluster in plot_data.groupby('Cluster'):   
        cluster.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
                     color = cmap((i)*1.0/(len(centers)-1)), label = 'Cluster %i'%(i), s=0.1);

    # Plot centers with indicators
    for i, c in enumerate(centers):
        ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', \
                   alpha = 1, linewidth = 2, marker = 'o', s=200);
        ax.scatter(x = c[0], y = c[1], marker='$%d$'%(i), alpha = 1, s=50);

    # Plot transformed sample points 
    #ax.scatter(x = pca_samples[:,0], y = pca_samples[:,1], \
               #s = 150, linewidth = 4, color = 'black', marker = 'x');

    # Set plot title
    ax.set_title("Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\n");

###################################### Plot Clustering #####################################
cluster_results(reduced_data,preds=preds,centers=centers)
#plt.show()

#####################################Data recovery #########################################


true_centers=pca.inverse_transform(centers)
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = new_df1.keys())
true_centers.index = segments
display(true_centers)

#true_centers.to_excel('new centers without skintype k=5 without 2010.xlsx')

################################ Track back to Original Reduced Data ####################################

df_scaled_good_data['segment#']=preds

df_final=pd.concat([df[['CustomerId']],df_scaled_good_data],axis=1)
df_final.dropna(axis=0,inplace=True)

#df_final.to_csv('customer_data_with_segmentation5_no2010.csv')


