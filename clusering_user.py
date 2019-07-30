import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import prince

class UserCluster:
    def turn_to_matrix(self, X):
        vec = np.zeros((len(X), 300))
        for i in range(len(X)):
            vec[i] = np.array(X[i])
        return vec


    def main(self):
        df = pd.read_json('FacebookGraph/final_data_euclide.json')
        X = df.drop(['group', 'group_euclide','group_cosine', 'group_count', 'id','like_euclide', 'like_cosine', 'like_count','home','location','hom_lon','hom_lat','loc_lon', 'loc_lat'], axis=1)
        print(X.columns)

        mca = prince.MCA(
            n_components=2,
            n_iter=5,
            copy=True,
            check_input=True,
            engine='auto',
            random_state=42
        )
        X = [np.array(x) for x in X['like']]
        # X = np.concatenate(X, axis=0)

        X = mca.fit_transform(X)
        # ax = mca.plot_coordinates(
        #     X=X,
        #     ax=None,
        #     figsize=(6, 6),
        #     show_row_points=True,
        #     row_points_size=10,
        #     show_row_labels=False,
        #     show_column_points=True,
        #     column_points_size=30,
        #     show_column_labels=False,
        #     legend_n_cols=1
        # )
        # ax.get_figure().savefig('images/mca_coordinates.svg')

        wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            km.fit(X.values)
            wcss.append(km.inertia_)
        plt.plot(range(1,11), wcss)
        plt.show()

        X = X.values
        # km2 = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
        km5 = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
        # y_means = km5.fit_predict(X_trans.values)
        y_means = km5.fit_predict(X)
        plt.scatter(X[y_means==0,0],X[y_means==0,1],s=50, c='purple',label='Cluster1')
        plt.scatter(X[y_means==1,0],X[y_means==1,1],s=50, c='blue',label='Cluster2')
        plt.scatter(X[y_means==2,0],X[y_means==2,1],s=50, c='green',label='Cluster3')
        plt.scatter(X[y_means==3,0],X[y_means==3,1],s=50, c='cyan',label='Cluster4')
        # plt.scatter(X[y_means==4,0],X[y_means==4,1],s=50, c='red',label='Cluster5')
        # plt.scatter(X[y_means==5,0],X[y_means==5,1],s=50, c='black',label='Cluster6')
        # plt.scatter(X[y_means==6,0],X[y_means==6,1],s=50, c='orange',label='Cluster7')
        
        plt.show()

if __name__=="__main__":
    a = UserCluster()
    a.main()