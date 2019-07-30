import urllib3
import facebook
import requests
import pickle as pkl
import json, csv, sys

class GetData:
    def __init__(self, dirpath, page_id, limit_num_users):
        self.access_token = "EAAAAAYsX7TsBAEjBuzX4dcYyRUkQt1C94n6FqBomE5j2ooXc8PJu2gNspZA5SMljGvaPcqjMeR5XTAbmmFpQC3r2pQsAmarfRWfS9OZBolIkqOGhjJYSiCUbREVTGtiEgUB3ZAiSYM8bGXh7gyZAOVlZATyIYXB7ODtg24aWsrwZDZD"
        self.graph = facebook.GraphAPI(access_token=self.access_token, version=3.1)
        self.likes = {}
        self.groups = {}
        self.location = {}
        self.hometown = {}
        self.user_ids = []
        self.dirpath = dirpath
        self.group_id = group_id
        self.limit_num_users = limit_num_users

    def get_user_likes(self, user_id):
        like_user = []
        try:
            likes_data = self.graph.get_object('/%s/likes' % user_id)
            while(True):
                try:
                    try:
                        for like in likes_data['data']:
                            like_user.append(like['id'])
                        likes_data = requests.get(likes_data['paging']['next']).json()
                    except KeyError:
                        break
                except Exception:
                    break
        except Exception:
            pass
        self.likes[user_id] = like_user
    
    def get_user_groups(self, user_id):
        group_user = []
        try:
            groups_data = self.graph.get_object('/%s/groups' % user_id)
            while(True):
                try:
                    try:
                        for group in groups_data['data']:
                            group_user.append(group['id'])
                        groups_data = requests.get(groups_data['paging']['next']).json()
                
                    except KeyError:
                        break
                except Exception:
                    break
        except Exception:
            pass
        self.groups[user_id] = group_user
    
    def get_user_location(self, user_id):
        try:
            try:
                location_data = self.graph.get_object('%s?fields=location' % user_id)['location']['name']
                self.location[user_id] = location_data
                del location_data
            except KeyError:
                pass
        except facebook.GraphAPIError:
            pass

    def get_user_hometown(self, user_id):
        try:
            try:
                hometown_data = self.graph.get_object('%s?fields=hometown' % user_id)['hometown']['name']
                self.hometown[user_id] = hometown_data
                del hometown_data
            except KeyError:
                pass
        except facebook.GraphAPIError:
            pass
    
    def get_user_id_from_groups(self):
        members_data = self.graph.get_object('/%s/members' % self.group_id)
        c = 0
        while (True):
            try:
                for member in members_data['data']:
                    user_id = member['id']
                    print(c)
                    if user_id not in self.user_ids:
                        self.user_ids.append(user_id)
                        c += 1
                if len(self.user_ids) <= self.limit_num_users:
                    members_data = requests.get(members_data['paging']['next']).json()      
                else: 
                    break
            except KeyError:
                break
        self.dump_to_pickle('user_ids.pkl', self.user_ids)
    
    def dump_to_pickle(self, filename, file):
        with open(filename, 'wb') as f:
            pkl.dump(file, f)
        f.close()

    def open_pickle_file(self, filename):
        with open(filename, 'rb') as f:
            data = pkl.load(f)
        f.close()
        return data

    def main(self):
        # self.get_user_id_from_groups()
        with open('user_ids.pkl', 'rb') as f:
            user_ids = pkl.load(f)
        f.close()
        count = 0
        # print(user_ids[2266])
        for i in range(2000,2500):
            user_id = user_ids[i]
            print(user_id)
            self.get_user_likes(user_id)
            self.get_user_groups(user_id)
            self.get_user_location(user_id)
            self.get_user_hometown(user_id)
            count += 1
            print(count)
            
        self.dump_to_pickle('data/likes2500.pkl', self.likes)
        self.dump_to_pickle('data/groups2500.pkl', self.groups)
        self.dump_to_pickle('data/location2500.pkl', self.location)
        self.dump_to_pickle('data/hometown2500.pkl', self.hometown)

        

if __name__=="__main__":
    dirpath  = 'rec'
    group_id = "870665749718859"
    limit_num_users = 6000
    a = GetData(dirpath, group_id, limit_num_users)
    a.main()