from __future__ import print_function
import code
import sqlite3
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from multiprocessing.dummy import Pool as ThreadPool

limit = 10000
cv = 10

'''
clf = Pipeline([('vect', TfidfVectorizer()),
                ('clf', SVC(kernel='linear', C=1)),
               ])
'''

vect = TfidfVectorizer()
sel = SelectFromModel(LinearSVC(loss='squared_hinge', penalty="l1", dual=False))
rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)

clf = Pipeline([('vect', vect),
                ('feature_selection', sel),
                ('clf', rfc)
               ])

conn = sqlite3.connect('twitter.db')
c = conn.cursor()
d = conn.cursor()
limit = (limit, )

def getFeatureVector(uid):
    con2 = sqlite3.connect('twitter.db')
    d = con2.cursor()
    d.execute('select group_concat(text) from tweet, user where tweet.user = user.id and user.id = ?;', uid)
    return d.fetchone()[0]

c.execute('select count(*) from tweet, user where user.is_bot = 1 and tweet.user = user.id;')
bot_count = c.fetchone()
c.execute('select count(*) from tweet, user where user.is_bot = 0 and tweet.user = user.id;')
user_count = c.fetchone()
print('bots: ' + str(bot_count[0]) + ' users: ' + str(user_count[0]))

print("Processing positive data: ", end="")
pool = ThreadPool(4)
pos_uids = c.execute('select id from user where user.is_bot = 1 order by random() limit ?;', limit)
pos_data = pool.map(getFeatureVector, pos_uids)
pool.close()
pool.join()
y_data_labels = np.ones(len(pos_data))
print("Done")

print("Processing negative data: ", end="")
pool = ThreadPool(4)
neg_uids = c.execute('select id from user where user.is_bot = 0 order by random() limit ?;', limit)
neg_data = pool.map(getFeatureVector, neg_uids)
pool.close()
pool.join()
y_data_labels = np.append(y_data_labels, np.zeros(len(neg_data)))
print("Done")

x_data_train = pos_data + neg_data

print("Fitting model: ", end="")
clf.fit_transform(x_data_train, y_data_labels)
print("Done")

scores = cross_validation.cross_val_score(clf, x_data_train, y_data_labels, cv=cv, scoring='f1_weighted')

print("scores: " + str(np.average(scores)))
print(clf.predict([getFeatureVector((22933636,))]))

importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]

feature_names = vect.get_feature_names()
table = []

for f in range(len(importances)):
    table.append([feature_names[indices[f]], importances[indices[f]]])

df = pd.DataFrame(table)
print(df.to_string(header=False))

code.interact(local=locals())
