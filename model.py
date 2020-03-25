from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class EASE:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'user_id'])
        items = self.item_enc.fit_transform(df.loc[:, 'item_id'])
        return users, items

    def fit(self, df, lambda_: float = 0.5, implicit=True):
        """
        df: pandas.DataFrame with columns user_id, item_id and (rating)
        lambda_: l2-regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """
        users, items = self._get_users_and_items(df)
        values = np.ones(df.shape[0]) if implicit else df['rating'].to_numpy() / df['rating'].max()

        X = csr_matrix((values, (users, items)))
        self.X = X

        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = X.dot(B)

    def predict(self, users, items, k, train=None):
        df = pd.DataFrame()
        users = self.user_enc.transform(users)
        items = self.item_enc.transform(items)
        for user in users:
            if train is None:
                watched = []
            else:
                watched = set(train.loc[train['user_id'] == user, 'item_id'])
            candidates = [item for item in items if item not in watched]
            pred = np.take(self.pred[user, :], candidates)
            res = np.argsort(pred)[::-1][:k]
            r = pd.DataFrame({
                "user_id": self.user_enc.inverse_transform([user] * len(res)),
                "item_id": self.item_enc.inverse_transform(np.take(candidates, res)),
                "score": np.take(pred, res)
            })
            df = df.append(r, ignore_index=True)
        return df
