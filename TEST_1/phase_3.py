""" 
Created by Alexis Kaldany for NLP @ GWU, for Prof. Jafari
Fall 2022
"""


import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,normalize
import spacy 
from sklearn.metrics import f1_score 



PATH = os.getcwd() + '/TEST_1/'
print(PATH)
train = pd.read_csv(PATH+'Train.csv')
test = pd.read_csv(PATH+ 'Test.csv')

print(train.shape)
print(test.shape)
print(train.head())
print(train.columns)

target = 'urgency'
print(train[target].value_counts())
categorical_features = ['ticket_type','category','sub_category1','sub_category2','business_service','impact']
text_features = ['title','body']
non_categoricals = ['title','body','urgency']
# target_labels = train[target].to_list()
# target_labels = [str(x) for x in set(target_labels)]
# print(target_labels)
target_labels = ['0','1','2','3']
# Drop the index and target
train_x = train.drop([target], axis=1)
test_x = test.drop([target], axis=1)

# Tokenize title and body
import spacy
nlp = spacy.load('en_core_web_sm')

# train_title = [nlp(text) for text in train_x['title']]
# train_body = [nlp(text) for text in train_x['body']]
# test_title = [nlp(text) for text in test_x['title']]
# test_body = [nlp(text) for text in test_x['body']]

"""
I decided to combine title and body into one column for easy of tokenization and vectorization and SVD
"""
combined_train = [str(train_x['title'].to_list()[i])+'.' +str(train_x['body'].to_list()[i]) for i in range(len(train_x))]
combined_test = [str(test_x['title'].to_list()[i])+'.' +str(test_x['body'].to_list()[i]) for i in range(len(test_x))]
combined_text_train = [nlp(text) for text in combined_train]
combined_text_test = [nlp(text) for text in combined_test]

## Preprocessing Categorical Features 
"""
I just normalized the categorical features as they are all ordinal and have no inherent meaning
"""
# def one_hot_encoding_categorical_features(df, categorical_features):
#     """
#     One hot encoding for categorical features
#     """
#     df = pd.get_dummies(df, columns=categorical_features)
#     return df

train_x_cats = train_x[categorical_features]
test_x_cats = test_x[categorical_features]

normalized_train_x_cats = normalize(train_x_cats)
normalized_test_x_cats = normalize(test_x_cats)

""" 
One hot encoding the categorical variables makes the array very wide and sparse, also it wasn't getting good results
"""
# train_x_cats = one_hot_encoding_categorical_features(train_x.drop(non_categoricals,axis=1), categorical_features)
# test_x_cats = one_hot_encoding_categorical_features(test_x.drop(non_categoricals,axis=1), categorical_features)

## Creating tidf matrix for title and body
tfidf= TfidfVectorizer()
tfidf.fit(combined_train)
train_x_transformed = tfidf.transform(combined_train).toarray()
test_x_transformed = tfidf.transform(combined_test).toarray()
print(train_x_transformed.shape)


""" 
SVD for dimensionality reduction sounds like a good idea but I never got a better result using it
"""
## Using SVD on the tfidf matrix to reduce the dimensionality
# from sklearn.decomposition import TruncatedSVD
# SVD= TruncatedSVD(n_components=300, n_iter=25, random_state=42).fit(train_x_transformed)
# SVD_train = SVD.transform(train_x_transformed)
# SVD_test = SVD.transform(test_x_transformed)


## Combining the categorical and text features
x_train_tfid_cats = pd.concat([pd.DataFrame(train_x_transformed),pd.DataFrame(normalized_train_x_cats)],axis=1)
x_test_tfid_cats = pd.concat([pd.DataFrame(test_x_transformed),pd.DataFrame(normalized_test_x_cats)],axis=1)

# print(x_train_tfid_cats.shape)
# print(x_train_tfid_cats.iloc[0,:])




## Testing Naive Bayes

# clf = MultinomialNB().fit(x_train_tfid_cats, train[target].values)
# predicted_nb = clf.predict(x_test_tfid_cats)

# report_nb = metrics.classification_report(test[target], predicted_nb, target_names=target_labels)
# # confusion_matrix =metrics.confusion_matrix(test[target], predicted)
# print(report_nb)

## Testing Logistic Regression
""" 
My model generated better outcomes using Logistic Regression
"""
clf = LogisticRegression(random_state=0,multi_class='multinomial').fit(x_train_tfid_cats, train[target].values)
predicted_lr = clf.predict(x_test_tfid_cats)

report_lr = metrics.classification_report(test[target], predicted_lr, target_names=target_labels,zero_division=0)
print(report_lr)
print(f1_score(test[target], predicted_lr, average='weighted'))