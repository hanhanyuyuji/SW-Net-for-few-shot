import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import numpy as np
import torch
import pandas as pd


fileObject = open('mini_meta_TCGA_60.pickle', 'rb')
data_episode1 = pickle.load(fileObject)

print(len(data_episode1.keys()))

print(data_episode1.keys())
#
# final_Support = np.array([])
# for key, val in data_episode1.items():
#     X = val['xs']
#     y= val['xs_class']
#     sel = VarianceThreshold(threshold=5)
#     X_new = sel.fit_transform(X)
#     support_X = sel.get_support(indices=True)
#     final_Support=np.union1d(final_Support, support_X)
#     #print(final_Support.shape)
# print(final_Support.shape)


# for key, val in data_episode1.items():
#     X = val['xs']
#     y= val['xs_class']
#     X_np=X.numpy()
#     X_df=pd.DataFrame(X_np)
#     fileName="CSVS/"+key+"xs.csv"
#     X_df.to_csv(fileName)
#     y_np=y.numpy()
#     y_df=pd.DataFrame(y_np)
#     fileName2 = "CSVS/" + key + "xs_class.csv"
#     y_df.to_csv(fileName2)





#
# final_Support2 = np.array([])
# for key, val in data_episode1.items():
#     X = val['xq']
#     y= val['xq_class']
#     sel = VarianceThreshold(threshold=4.983)
#     X_new = sel.fit_transform(X)
#     support_X = sel.get_support(indices=True)
#     final_Support2=np.union1d(final_Support2, support_X)
#     #print(final_Support.shape)





#print(data_episode1)
print(data_episode1['_EVENT-BLCA-0'])
print(data_episode1['_EVENT-BLCA-0']['xs'].shape)
print(len(data_episode1['_EVENT-BLCA-0']['xs']))
print(data_episode1['_EVENT-BLCA-0']['xs_class'])
print(data_episode1['_EVENT-BLCA-0']['xq'].shape)
print(data_episode1['_EVENT-BLCA-0']['xq_class'].shape)
#X=data_episode1['_EVENT-BLCA-0']['xs']
#print(X.shape)

# lsvc = LinearSVC(C=0.5, penalty="l1", dual=False).fit(X, y)
# model = SelectFromModel(lsvc, prefit=True)
#X_new = model.transform(X)
# X=data_episode1['_EVENT-BLCA-0']['xs']
# sel = VarianceThreshold(threshold=5)
# X_new=sel.fit_transform(X)
# support=sel.get_support(indices=True)
# print(support.shape)
# print(support)
# print(X_new.shape)
# print(X_new)

#X_new = SelectKBest(chi2, k=200).fit_transform(X, y)
# for key, value in data_episode1.items():
#     value['xs'];
#     value['xs_class']
#print(X_new.shape)