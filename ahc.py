import pickle
from sklearn.cluster import AgglomerativeClustering
import editdistance as edist
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

with open('sentences.pkl','rb') as f:
    content_list = pickle.load(f)
f.close()
# content_list = content_list[:100]
def lev_metrics(x,y):
    return int(edist.eval(content_list[int(x[0])], content_list[int(y[0])]))

X = np.arange(len(content_list)).reshape(-1,1)

m = pairwise_distances(X, X, metric=lev_metrics)
print (m)
agg = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='average')
u = agg.fit_predict(m)
print (list(u.labels_))