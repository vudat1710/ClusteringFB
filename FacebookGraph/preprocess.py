import pickle, os
from geopy.geocoders import Nominatim
from operator import itemgetter
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

class PreProcess:
    def __init__(self):
        self.city = {}
        self.like = {}
        self.group = {}

    def combine_data(self, num):
        likes = self.open_pickle('data/likes%d.pkl' % num)
        hometown = self.open_pickle('data/hometown%d.pkl' % num)
        groups = self.open_pickle('data/groups%d.pkl' % num)
        location = self.open_pickle('data/location%d.pkl' % num)
        dfhome = pd.DataFrame(list(hometown.items()), columns=['id','home'])
        dflike = pd.DataFrame(list(likes.items()), columns=['id','like'])
        dfgroup = pd.DataFrame(list(groups.items()), columns=['id','group'])
        dfloc = pd.DataFrame(list(location.items()), columns=['id','location'])
        t1 = pd.merge(dflike, dfgroup, on='id',how='left')
        t2 = pd.merge(t1, dfhome, on='id',how='left')
        res = pd.merge(t2, dfloc, on='id', how='left')
        res.to_json('data%d.json' % num, orient='records')

    def open_pickle(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        f.close()
        return data
    
    def get_lon_lat(self, hometown, location):
        geo = Nominatim(user_agent='myapp')
        for city_name in hometown:
            if city_name is not None:
                city_name = city_name.strip()
                if city_name not in self.city.keys():
                    print(city_name)
                    try:
                        loc = geo.geocode(city_name)
                        if loc is not None:
                            self.city[city_name] = [float(loc.raw['lat']), float(loc.raw['lon'])]
                            print(self.city[city_name])
                        else:
                            self.city[city_name] = [0,0]
                            print(self.city[city_name])
                    except Exception:
                        self.city[city_name] = [0,0]
                        print(self.city[city_name])
        for city_name in location:
            if city_name is not None:
                city_name = city_name.strip()
                if city_name not in self.city.keys():
                    print(city_name)
                    try:
                        loc = geo.geocode(city_name)
                        if loc is not None:
                            self.city[city_name] = [float(loc.raw['lat']), float(loc.raw['lon'])]
                            print(self.city[city_name])
                        else:
                            self.city[city_name] = [0,0]
                            print(self.city[city_name])
                    except Exception:
                        pass
        print (len(self.city))
        with open('city_lon_lat.pkl', 'wb') as f:
            pickle.dump(self.city, f)
        f.close()
    
    def turn_to_lon_lat(self, x):
        geo = Nominatim(user_agent='myapp')
        if x is not None:
            if x in self.city.keys():
                print(self.city[x])
                return self.city[x]
            else:
                try:
                    loc = geo.geocode(x)
                    if loc is not None:
                        print([float(loc.raw['lat']), float(loc.raw['lon'])])
                        return [float(loc.raw['lat']), float(loc.raw['lon'])]
                    else:
                        return [0,0]
                except Exception:
                    return [0,0]
        else:
            return [0,0]
    
    def add_lon_lat_column(self, df):
        df['home'] = df['home'].apply(lambda x: self.turn_to_lon_lat(x))
        df['location'] = df['location'].apply(lambda x: self.turn_to_lon_lat(x))
    
    def get_vector(self, x, _list):
        vec = np.zeros(300)
        for i in range(len(x)):
            id = x[i]
            if id in _list:
                vec[i] = 1.0
        return vec

    def turn_to_vector(self, df):
        df['like'] = df['like'].apply(lambda x: self.get_vector(x, self.like))
        df['group'] = df['group'].apply(lambda x: self.get_vector(x, self.group))
            
    def get_300_first_each(self,x):
        if len(x) <= 300:
            return x
        else:
            _dict = {}
            for id in x:
                if id in _dict:
                    _dict[id] += 1
                else:
                    _dict[id] = 1
            _dict = sorted(_dict.items(), key = itemgetter(1), reverse=True)
            _dict = _dict[:300]
            _dict = [x[0] for x in _dict]
            return _dict
    
    def turn_300_first(self, df):
        df['like_count'] = df['like'].apply(lambda x: len(x))
        df['group_count'] = df['group'].apply(lambda x: len(x))
        df['like'] = df['like'].apply(lambda x: self.get_300_first_each(x))
        df['group'] = df['group'].apply(lambda x: self.get_300_first_each(x))
    
    def get_popular_pages_groups(self, df):
        likes_data = list(df['like'])
        groups_data = list(df['group'])
        print('Likes data:')
        for i in range(len(likes_data)):
            for like in likes_data[i]:
                if like not in self.like:
                    self.like[like] = 1
                else:
                    self.like[like] += 1
        print('Groups data:')
        for j in range(len(groups_data)):
            for group in groups_data[j]:
                if group not in self.group:
                    self.group[group] = 1
                else:
                    self.group[group] += 1
        likes = sorted(self.like.items(), key = itemgetter(1), reverse=True)
        groups = sorted(self.group.items(), key = itemgetter(1), reverse=True)
        with open('top_likes.pkl', 'wb') as f:
            pickle.dump(likes[:300], f)
        f.close()
        with open('top_groups.pkl', 'wb') as f:
            pickle.dump(groups[:300], f)
        f.close()
    
    def concat_json(self, DIRECTORY):
        data = []
        for filename in os.listdir(DIRECTORY):
            if filename.endswith(".json"):
                print ("%s" % filename)
                filepath = "%s%s" %(DIRECTORY, filename)
                print(filepath)
                df = pd.read_json(filepath)
                data.append(df)
                print("done %s" % filepath)
        print (len(data))
        df = pd.concat(data, axis=0)
        df.to_json('data.json', orient = 'records')
    
    def cosine(self, x):
        base = np.ones(300).reshape(1,-1)
        x = np.array(x).reshape(1,-1)
        return cosine_similarity(base,x)[0][0]

    def add_cosine_col(self, df):
        df['like_cosine'] = df['like'].apply(lambda x: self.cosine(x))
        df['group_cosine'] = df['group'].apply(lambda x: self.cosine(x))

    def seperate_data(self, df):
        df['loc_lon'] = df['location'].apply(lambda x: x[0])
        df['loc_lat'] = df['home'].apply(lambda x: x[1])
        df['hom_lon'] = df['location'].apply(lambda x: x[0])
        df['hom_lat'] = df['home'].apply(lambda x: x[1])
    
    def euclide(self,x):
        base = np.ones(300).reshape(1,-1)
        x = np.array(x).reshape(1,-1)
        return euclidean_distances(base, x)[0][0]
    
    def add_euclide_col(self, df):
        df['like_euclide'] = df['like'].apply(lambda x: self.euclide(x))
        df['group_euclide'] = df['group'].apply(lambda x: self.euclide(x))

    def main(self):
        # self.combine_data(2500)
        # self.concat_json('data_not_geo/')

        df = pd.read_json('data.json')
        # location = list(df['location'])
        # hometown = list(df['home'])
        # self.get_lon_lat(hometown, location)

        self.city = self.open_pickle('city_lon_lat.pkl')
        self.add_lon_lat_column(df)
        # df.to_json('data_geo.json', orient='records')
        self.turn_300_first(df)
        # df.to_json('data_300_first.json', orient='records')
        self.get_popular_pages_groups(df)
        self.like = self.open_pickle('top_likes.pkl')
        self.group = self.open_pickle('top_groups.pkl')
        self.like = [x[0] for x in self.like]
        self.group = [x[0] for x in self.group]
        self.turn_to_vector(df)
        # df.to_json('final_data.json', orient='records')
        self.add_cosine_col(df)
        # df.to_json('final_data_cosine', orient='records')
        self.add_euclide_col(df)
        df.to_json('final_data_euclide.json', orient='records')




if __name__=="__main__":
    a = PreProcess()
    a.main()

