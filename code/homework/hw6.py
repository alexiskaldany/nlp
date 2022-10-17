
from turtle import back
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import os
import json

# =================================================================
# Class_Ex1:
#  Use the following datframe as the sample data.
# Find the conditional probability of Char given the Occurrence.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')
df = pd.DataFrame({'Char':['f', 'b', 'f', 'b','f', 'b', 'f', 'f'], 'Occurance':['o1', 'o1', 'o2', 'o3','o2', 'o2', 'o1', 'o3'], 'C':np.random.randn(8), 'D':np.random.randn(8)})

Occurance_value_count = df['Occurance'].value_counts()
Char_value_count = df['Char'].value_counts()
Char_given_Occurance = df.groupby(['Occurance', 'Char']).size().unstack(fill_value=0)
Char_given_Occurance['P(b|Occurance)'] = Char_given_Occurance['b'] / Occurance_value_count
Char_given_Occurance['P(f|Occurance)'] = Char_given_Occurance['f'] / Occurance_value_count


print(20*'-' + 'End Q1' + 20*'-')

# =================================================================
# Class_Ex2:
# Use the following datframe as the sample data.
# Find the conditional probability occurrence of thw word given a sentiment.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')

df1 = pd.DataFrame({'Word': ['Good', 'Bad', 'Awesome', 'Beautiful', 'Terrible', 'Horrible'],
                     'Occurrence': ['One', 'Two', 'One', 'Three', 'One', 'Two'],
                     'sentiment': ['P', 'N', 'P', 'P', 'N', 'N'],})

occurance = df1['Occurrence'].to_dict()











print(20*'-' + 'End Q2' + 20*'-')
#%%
# =================================================================
# Class_Ex3:
# Read the data.csv file.
# Answer the following question
# 1- In this dataset we have a lot of responses in text and each response has a label.
# 2- Our goal is to correctly model the texts into its label.
# Hint: you need to read the text responses and perform preprocessing on it.
# such as normalization, legitimation, cleaning, stopwords removal and POS tagging.
# then use any methods you learned in the lecture to convert each response into meaningful numbers.
# 3- Apply Naive bayes and look at appropriate evaluation metric.
# 4- Explain your results very carefully.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q3' + 20*'-')
print(os.getcwd())
data_path = os.getcwd() + '/code/homework/data5.csv'
print(data_path)
df = pd.read_csv(data_path,encoding_errors='ignore')
# df['label'] = df['label'].str.replace('__label__', '').astype(int)

# df_train = df.sample(frac=0.8, random_state=200)
# df_test = df.drop(df_train.index)

# tfidf= TfidfVectorizer()
# tfidf.fit(df_train['text'])
# X_train_tfidf =tfidf.transform(df_train['text'])


# clf = MultinomialNB().fit(X_train_tfidf, df_train['label'])
# X_test_tfidf = tfidf.transform(df_test['text'])
# predicted = clf.predict(X_test_tfidf)

# print(metrics.classification_report(df_test['label'], predicted, target_names=df['label'].unique()))
# print(metrics.confusion_matrix(df_test['label'], predicted))

def naive_bayes(dataframe:pd.DataFrame,feature:str,target:str):
    df_train = dataframe.sample(frac=0.8, random_state=200)
    df_test = dataframe.drop(df_train.index)
    print(df_train.shape)
    print(df_test.shape)
    tfidf= TfidfVectorizer()
    tfidf.fit(df_train[feature])
    X_train_tfidf =tfidf.transform(df_train[feature])
    print(X_train_tfidf.shape)
    clf = MultinomialNB().fit(X_train_tfidf, df_train[target].values)
    X_test_tfidf = tfidf.transform(df_test[feature])
    predicted = clf.predict(X_test_tfidf)
    report = metrics.classification_report(df_test[target], predicted, target_names=df[target].unique())
    confusion_matrix =metrics.confusion_matrix(df_test[target], predicted)
    return [report, confusion_matrix]

reports = naive_bayes(df,'text','label')
print(reports[0])








print(20*'-' + 'End Q3' + 20*'-')

# =================================================================
# Class_Ex4:
# Use Naive bayes classifier for this problem,
# Write a text classification pipeline to classify movie reviews as either positive or negative.
# Find a good set of parameters using grid search. hint: grid search on n gram
# Evaluate the performance on a held out test set.
# hint1: use nltk movie reviews dataset
# from nltk.corpus import movie_reviews

# ----------------------------------------------------------------

print(20*'-' + 'Begin Q4' + 20*'-')

from nltk.corpus import movie_reviews

positive = list(movie_reviews.sents(categories=['pos']))
# negative= movie_reviews.sents(categories=['neg'])
# review_list =[{'text': ' '.join(i), 'label': 'pos'} for i in positive]
# review_list.extend([{'text': ' '.join(i), 'label': 'neg'} for i in negative])

# with open('review_list.json', 'w') as f:
#     json.dump(review_list, f)
# df = pd.DataFrame(review_list)
# df_train = df.sample(frac=0.8, random_state=200)
# df_test = df.drop(df_train.index)
# from sklearn.feature_extraction.text import CountVectorizer

# vectorizer = CountVectorizer(ngram_range=(1, 100),token_pattern = r"(?u)\b\w+\b")
# X_train = vectorizer.fit_transform(df_train['text'])
# X_test = vectorizer.transform(df_test['text'])
# clf = MultinomialNB().fit(X_train, df_train['label'].values)
# predicted = clf.predict(X_test)
# print(metrics.classification_report(df_test['label'], predicted, target_names=df['label'].unique()))
# positive_list = [(" ".join(positive[i]), 'pos') for i in range(len(positive))]











print(20*'-' + 'End Q4' + 20*'-')

# =================================================================
# Class_Ex5:
# Calculate accuracy percentage between two lists
# calculate a confusion matrix
# Write your own code - No packages
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q5' + 20*'-')










print(20*'-' + 'End Q5' + 20*'-')
# =================================================================
# Class_Ex6:
# Read the data.csv file.
# Answer the following question
# 1- In this dataset we have a lot of responses in text and each response has a label.
# 2- Our goal is to correctly model the texts into its label.
# Hint: you need to read the text responses and perform preprocessing on it.
# such as normalization, legitimation, cleaning, stopwords removal and POS tagging.
# then use any methods you learned in the lecture to convert each response into meaningful numbers.
# 3- Apply Logistic Regression  and look at appropriate evaluation metric.
# 4- Apply LSA method and compare results.
# 5- Explain your results very carefully.

# ----------------------------------------------------------------
print(20*'-' + 'Begin Q6' + 20*'-')

data_path = os.getcwd()+'/code/homework/data5.csv'
dataframe = pd.read_csv(data_path,encoding_errors='ignore')

def logistic_regression(dataframe:pd.DataFrame,feature:str,target:str):
    from sklearn.linear_model import LogisticRegression
    df_train = dataframe.sample(frac=0.8, random_state=200)
    df_test = dataframe.drop(df_train.index)
    print(df_train.shape)
    print(df_test.shape)
    tfidf= TfidfVectorizer()
    tfidf.fit(df_train[feature])
    X_train_tfidf =tfidf.transform(df_train[feature])
    print(X_train_tfidf.shape)
    clf = LogisticRegression().fit(X_train_tfidf, df_train[target].values)
    X_test_tfidf = tfidf.transform(df_test[feature])
    predicted = clf.predict(X_test_tfidf)
    report = metrics.classification_report(df_test[target], predicted, target_names=df[target].unique())
    confusion_matrix =metrics.confusion_matrix(df_test[target], predicted)
    return [report, confusion_matrix]


reports = logistic_regression(df,'text','label')


# from sklearn.decomposition import TruncatedSVD
# df_train = dataframe.sample(frac=0.8, random_state=200)
# df_test = dataframe.drop(df_train.index)
# print(df_train.shape)
# print(df_test.shape)
# print(df_train.columns)

# tfidf= TfidfVectorizer()
# tfidf.fit(df_train['text'])
# X_train_tfidf =tfidf.transform(df_train['text'])
# print(X_train_tfidf.shape)
# X_train_svd = TruncatedSVD(n_components=100, n_iter=20, random_state=42).fit_transform(X_train_tfidf)
# ### Try different n_components and fitting to different models (linear, logistic, Bayes.)
# clf = LogisticRegression().fit(X_train_svd, df_train['label'].values)
# predicted = clf.predict(X_train_svd)
# report = metrics.classification_report(df_train['label'], predicted, target_names=df['label'].unique())
# confusion_matrix =metrics.confusion_matrix(df_train['label'], predicted)
# print(report)
# print(confusion_matrix)



# back_to_text = tfidf.inverse_transform(new_X_train_tfidf)
# print(len(back_to_text[0]))
# print(back_to_text[0])
# print(feature_names[0])

def with_svd_all_models(dataframe:pd.DataFrame,feature:str,target:str):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import metrics
    df_train = dataframe.sample(frac=0.8, random_state=200)
    df_test = dataframe.drop(df_train.index)
    tfidf= TfidfVectorizer()
    tfidf.fit(df_train[feature])
    X_train_tfidf =tfidf.transform(df_train[feature])
    X_test_tfidf = tfidf.transform(df_test[feature])
    SVD_fit= TruncatedSVD(n_components=1000, n_iter=25, random_state=42).fit(X_train_tfidf) 
    X_train_svd = SVD_fit.transform(X_train_tfidf)
    LR_PREDICT = LogisticRegression().fit(X_train_svd, df_train[target].values).predict(SVD_fit.transform(X_test_tfidf))
    report = metrics.classification_report(df_test[target], LR_PREDICT, target_names=df[target].unique())
    confusion_matrix =metrics.confusion_matrix(df_test[target], LR_PREDICT)
    # BAYES_PREDICT = MultinomialNB().fit(X_train_svd, df_train[target].values).predict(SVD_fit.transform(X_test_tfidf))
    # print('MultinomialNB')
    # print(metrics.classification_report(df_test[target], BAYES_PREDICT, target_names=df[target].unique()))
    return [report, confusion_matrix]
svd_reports = with_svd_all_models(df,'text','label')
print('Logistic Regression')
print(reports[0])
print('LR with SVD')
print(svd_reports[0])






print(20*'-' + 'End Q6' + 20*'-')

# =================================================================
# Class_Ex7:
# Use logistic regression classifier for this problem,
# Write a text classification pipeline to classify movie reviews as either positive or negative.
# Find a good set of parameters using grid search. hint: grid search on n gram
# Evaluate the performance on a held out test set.
# hint1: use nltk movie reviews dataset
# from nltk.corpus import movie_reviews
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q7' + 20*'-')












print(20*'-' + 'End Q7' + 20*'-')

# =================================================================
# Class_Ex8:
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q8' + 20*'-')










print(20*'-' + 'End Q8' + 20*'-')
# =================================================================
# Class_Ex9:
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q9' + 20*'-')










print(20*'-' + 'End Q9' + 20*'-')
# =================================================================
# Class_Ex10:
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q10' + 20*'-')











print(20*'-' + 'End Q10' + 20*'-')
# =================================================================
# Class_Ex11:
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q11' + 20*'-')











print(20*'-' + 'End Q11' + 20*'-')
# =================================================================
# Class_Ex12:
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q12' + 20*'-')












print(20*'-' + 'End Q12' + 20*'-')
# =================================================================
# Class_Ex13:
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q13' + 20*'-')









print(20*'-' + 'End Q13' + 20*'-')
# =================================================================
# Class_Ex14:
#

# ----------------------------------------------------------------
print(20*'-' + 'Begin Q14' + 20*'-')









print(20*'-' + 'End Q14' + 20*'-')

# =================================================================










# %%
