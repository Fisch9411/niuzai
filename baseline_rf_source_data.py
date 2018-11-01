'''
@author : Lucas
@time : 2018-10-30
inner peace is of most important
'''

import pandas as pd
import time

from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor

train_list, test_list = [], []

gl_map = {1: 10, 2: 10, 3: 40, 4: 50}

for i in range(1, 5):
    train_ = pd.read_csv('data/train_{}.csv'.format(i))
    col_add = pd.DataFrame({'装机功率': [gl_map[i]] * len(train_['时间'])})
    train_list.append(pd.concat([train_, col_add], axis=1))
    test_ = pd.read_csv('data/test_{}.csv'.format(i))
    col_add = pd.DataFrame({'装机功率': [gl_map[i]] * len(test_['时间'])})
    test_list.append(pd.concat([test_, col_add], axis=1))

train = pd.concat(train_list, axis=0)
train = train[train['实发辐照度'] >= 0]
test = pd.concat(test_list, axis=0)

# time feature
train['day'] = train['时间'].apply(
    lambda x: int(time.strftime('%d', time.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S'))))
# train['month'] = train['时间'].apply(
#     lambda x: int(time.strftime('%m', time.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S'))))
# test['month'] = test['时间'].apply(lambda x: int(time.strftime('%m', time.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S'))))
test['day'] = test['时间'].apply(lambda x: int(time.strftime('%d', time.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S'))))
train['hour'] = train['时间'].apply(
    lambda x: int(time.strftime('%H', time.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S'))))
test['hour'] = test['时间'].apply(lambda x: int(time.strftime('%H', time.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S'))))

features = ['时间', '辐照度', '风速', '风向', '温度', '压强', '湿度']

# prepare train data and test data
X_train = train.drop(['实际功率', '实发辐照度', '时间','风向'], axis=1).values
y_train = train.loc[:, '实际功率'].values
X_test = test.drop(['id', '时间','风向'], axis=1).values
res = test.loc[:, ['id']]

# random forest model
model_rf = RandomForestRegressor(random_state=1, n_estimators=20, min_samples_leaf=1, min_samples_split=2)
skf = list(StratifiedKFold(y_train, n_folds=5, shuffle=True, random_state=1024))
for i, (train_index, test_index) in enumerate(skf):
    print("Fold", i)
    model_rf.fit(X_train[train_index], y_train[train_index])
    test_pred = model_rf.predict(X_test)
    print('test mean:', test_pred.mean())
    res['prob_%s' % str(i)] = test_pred

# 加权平均
res['predicition'] = 0
for i in range(5):
    res['predicition'] += res['prob_%s' % str(i)]
res['predicition'] = res['predicition'] / 5


mean = res['predicition'].mean()
print('mean:', mean)
now = time.localtime(time.time())
now = time.strftime('%m-%d-%H-%M', now)
res[['id', 'predicition']].to_csv("./result/lgb_baseline_%s.csv" % now, index=False)
