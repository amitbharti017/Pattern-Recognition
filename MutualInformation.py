
import pandas as pd
import category_encoders as ce

from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif


data = fetch_openml(name='kdd_internet_usage')
df = data.frame
df.info()


target = 'Who_Pays_for_Access_Work'
y = df[target]
X_cat = data.data.drop(columns=['Who_Pays_for_Access_Dont_Know',
       'Who_Pays_for_Access_Other', 'Who_Pays_for_Access_Parents',
       'Who_Pays_for_Access_School', 'Who_Pays_for_Access_Self'])


encoder = ce.LeaveOneOutEncoder(return_df=True)
X = encoder.fit_transform(X_cat, y)


X.shape

selector = SelectKBest(mutual_info_classif, k=20)
X_reduced = selector.fit_transform(X, y)
X_reduced.shape

cols = selector.get_support(indices=True)
selected_columns = X.iloc[:,cols].columns.tolist()
selected_columns

selector = SelectPercentile(mutual_info_classif, percentile=25)
X_reduced = selector.fit_transform(X, y)
X_reduced.shape

cols = selector.get_support(indices=True)
selected_columns = X.iloc[:,cols].columns.tolist()
selected_columns
