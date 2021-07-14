from classifiers.base_classifier import BaseClassifier
import numpy as np
from scipy import sparse as sp
import pandas as pd
from scipy.sparse.linalg import norm
from sklearn import preprocessing


class UUCF_classifier(BaseClassifier):
    def __init__(self):
        super().__init__('UUCF')
        self.engagement_matrix = None
        self.tweetId_to_tweetIDX = None
        self.tweetIDX_to_tweetId = None
        self.userId_to_userIDX = None
        self.userIDX_to_userId = None

    def train(self, X, y, engagement):
        engaging_user_ids = X["engaging_user_id"].unique()
        engaged_user_ids = X["engaged_with_user_id"].unique()
        all_user_ids = np.unique(np.append(engaging_user_ids, engaged_user_ids))
        tweet_ids = X["tweet_id"].unique()

        ## create internal ids for movies and users, that have consecutive indexes starting from 0
        tweetId_to_tweetIDX = dict(zip(tweet_ids, range(0, tweet_ids.size)))
        tweetIDX_to_tweetId = dict(zip(range(0, tweet_ids.size), tweet_ids))

        self.tweetId_to_tweetIDX = tweetId_to_tweetIDX
        self.tweetIDX_to_tweetId = tweetIDX_to_tweetId

        userId_to_userIDX = dict(zip(all_user_ids, range(0, all_user_ids.size)))
        userIDX_to_userId = dict(zip(range(0, all_user_ids.size), all_user_ids))

        self.userId_to_userIDX = userId_to_userIDX
        self.userIDX_to_userId = userIDX_to_userId
        raw_data = X[X[engagement].notnull()]

        ratings = pd.concat(
            [raw_data['engaging_user_id'].map(userId_to_userIDX), raw_data['tweet_id'].map(tweetId_to_tweetIDX)],
            axis=1)
        # set feedback to 0 or 1
        ratings.columns = ['engaging_user_id', 'tweet_id']
        ratings["engagement"] = np.ones_like(ratings['engaging_user_id'].values, dtype=np.int8)
        R = sp.csr_matrix((ratings["engagement"], (ratings['engaging_user_id'], ratings['tweet_id'])))
        self.engagement_matrix = R

    def compute_user_similarities(self, u_id):
        R = self.engagement_matrix.copy()
        norm_R = preprocessing.normalize(R, axis=1)
        u = norm_R[u_id, :].copy()
        uU = norm_R.dot(u.T).toarray().flatten()
        return uU
        # R=self.engagement_matrix.copy()
        # norm_vector=norm((R),axis=1)
        # val = np.repeat(norm_vector, R.getnnz(axis=1))
        # np.divide(R_copy.data,val,out=R.data,where=(val!=0))
        # u=R[u_id,:].copy()
        # user_sim=R.dot(u.T)
        # user_sim_arrray=user_sim.A
        # uU=user_sim_arrray.ravel()
        # return uU

    def create_user_neighborhood(self, u_id, i_id):
        k = 5
        with_abs_sim = False
        nh = {}
        uU = self.compute_user_similarities(u_id)
        uU_copy = uU.copy()
        R_dok = self.engagement_matrix.todok()

        if (with_abs_sim):
            uU_copy = np.absolute(uU_copy)
        sort = np.argsort(uU_copy)[::-1]
        number_of_inserted_neighbors = 0
        for index in sort:
            if (index == u_id):
                continue
            if (index, i_id) in R_dok:
                nh[index] = uU[index]
                number_of_inserted_neighbors += 1
                if (number_of_inserted_neighbors == k):
                    break

        return nh

    def predict_engagement(self, u_id, i_id):
        userIDX = self.userId_to_userIDX.get(u_id)
        tweet_IDX = self.tweetId_to_tweetIDX.get(i_id)
        if userIDX is None or tweet_IDX is None:
            return np.random.randint(0, 2)
        neighborhood = self.create_user_neighborhood(userIDX, tweet_IDX)

        nbh_avg = 0
        similarity_sum = 0.5
        R = self.engagement_matrix.copy()

        for i in neighborhood:
            nbh_avg = nbh_avg + neighborhood[i] * R[i, tweet_IDX]
            similarity_sum = similarity_sum + abs(neighborhood[i])

        nbh_avg = nbh_avg / similarity_sum
        prediction = nbh_avg

        return prediction

    def predict_proba(self, X_test):
        predictions = []
        for index, row in X_test.iterrows():
            u_id = row["engaging_user_id"]
            i_id = row['tweet_id']
            prediction = self.predict_engagement(u_id, i_id)
            prediction=round(prediction)
            predictions.append(prediction)
            if (len(predictions) % 1000) == 0:
                print(len(predictions))

        return predictions
