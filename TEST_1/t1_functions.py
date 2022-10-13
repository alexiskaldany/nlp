# String and Files
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics

def count_repeated_characters(input_string:str):
    result = {}
    for c in input_string:
        if c in result:
            result[c] += 1
        else:
            result[c] = 1
    return result

def read_first_n_lines(file_name:str, n:int):
    with open(file_name, "r") as f:
        n_lines = f.readlines()[:n]
        combined_lines = "".join(n_lines)
        print(max(combined_lines.split(), key=len))
        results = {}
        for x in combined_lines.split():
            if x in results.keys():
                results[x] += 1
            else:
                results[x] = 1
        print(results)
        
      
# NLTK

def get_lemma(tokens)-> list:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    
### Models

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