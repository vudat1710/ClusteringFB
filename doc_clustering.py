import json, pickle
import pandas as pd 
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from pyvi import ViTokenizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.cluster.hierarchy import ward, dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import editdistance as edist
from sklearn.metrics.pairwise import pairwise_distances

class DocClustering:
    def __init__(self, filepath):
        self.filepath = filepath

    def get_df_from_file(self):
        # df = pd.read_json(self.filepath)
        df = pd.read_json('~/Downloads/Training OSP/Training week 1-2/Fresher_Data_Science_week1-2/Week 1/Crawler/baomoicrawler/data_week_5.json')
        temp_df = df.groupby(df['topic']).count()
        drop_topic = list(temp_df.loc[temp_df['content'] < 20].index)
        drops = []
        for topic in drop_topic:
            drops.extend(list(df.loc[df['topic'] == topic].index.values))
        df = df.drop(drops, axis=0).reset_index(drop=True)
        df['len'] = df['content'].apply(lambda x: len(x))
        drops_0 = list(df.loc[df['len'] == 0].index)
        df = df.drop(drops_0, axis=0).reset_index(drop=True)
        return df
    
    def get_content_list(self, df):
        return df['content'].tolist()
    
    def get_topic_list(self, df):
        return df['topic'].unique().tolist()
    
    def get_stopwords_punc(self, filepath):
        stop_words = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                stop_words.append(line.strip())
        f.close()
        punc = list(punctuation)
        stop_words = stop_words + punc
        return stop_words

    def get_word_from_contents(self, df):
        sentences = []
        content_list = self.get_content_list(df)
        stopwords = self.get_stopwords_punc('stopwords.txt')
        for content in content_list:
            sent = []
            word_sent = ViTokenizer.tokenize(content.lower())
            for word in word_sent.split(' '):
                if word not in stopwords:
                    if '_' in word or word.isalpha():
                        sent.append(word)
            sentences.append(' '.join(sent))
            del sent
        with open('sentences_2.pkl', 'wb') as f:
            pickle.dump(sentences, f)
        f.close()
    
    def get_vocab(self,content_list):
        total_vocab = []
        for content in content_list:
            word_list = content.split(' ')
            total_vocab.extend(word_list)
        return total_vocab

    
    def main(self):
        df = self.get_df_from_file()
        topic_list = self.get_topic_list(df)
        true_k = len(topic_list)
        # self.get_word_from_contents(df)
        vectorizer = TfidfVectorizer(max_df=0.8)
        with open('sentences.pkl','rb') as f:
            content_list = pickle.load(f)
        f.close()
        # content_list = content_list[:1000]
        X = vectorizer.fit_transform(content_list)
        

        
        model = KMeans(n_clusters=true_k)
        model.fit(X)
        with open('model_kmeans.pkl', 'wb') as f:
            pickle.dump(model, f)
        f.close()
        order_centroids = model.cluster_centers_.argsort()[:,::-1]
        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print ("Cluster %d:" % i)
            for ind in order_centroids[i,:10]:
                print(' %s' % terms[ind])    
        df['kmeans'] = pd.DataFrame(model.labels_)
        df.to_json('kmeans_2.json', orient='records')

        
        model = AgglomerativeClustering(n_clusters=true_k, affinity='euclidean', linkage='ward')
        # model = AgglomerativeClustering(n_clusters=true_k, affinity='cosine', linkage='average')
        model.fit(X.toarray())
        df['ahc'] = pd.DataFrame(model.labels_)
        df.to_json('ahc_2.json', orient='records')



        # with open('model_kmeans.pkl', 'rb') as f:
        #     model = pickle.load(f)
        # f.close()

        # with open('vectorizer.pkl', 'rb') as f:
        #     vectorizer = pickle.load(f)
        # f.close()
        # X = vectorizer.transform(content_list)

#         sent_transform = vectorizer.transform(sentence)
#         print(model.predict(sent_transform))


if __name__=="__main__":
    filepath = 'data_week_5.json'
    a = DocClustering(filepath)
    a.main()