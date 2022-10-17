# String and Files
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import spacy 
from collections import Counter, defaultdict

nlp = spacy.load("en_core_web_sm")
def stats_dict(text):
        """
        Generate text summary
        """
        stat_dict = {}
        stat_dict["text_length"] = len(text)
        stat_dict["text_sentences"] = len(list(nlp(text).sents))
        stat_dict["text_words"] = len(text.split(" "))
        stat_dict["text_lines"] = len(text.splitlines())
        stat_dict["text_unique_words"] = len(set(text.split(" ")))
        stat_dict["text_unique_words_percentage"] = round((stat_dict["text_unique_words"]/stat_dict["text_words"])*100,2)
        return stat_dict

def word_counter(text):
        """
        Count words in text
        """
        word_counter = defaultdict(int)
        for word in text.split(" "):
            if word in word_counter:
                word_counter[word] += 1
            else:
                word_counter[word] = 1
        word_counter = {k: v for k, v in sorted(word_counter.items(), key=lambda item: item[1],reverse=True)}
        return word_counter

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
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas
from nltk.corpus import brown
def get_conditional_frequency_distribution(dataframe:pd.DataFrame,feature:str,target:str):
    cfd = nltk.ConditionalFreqDist(
        (target, feature)
        for target in dataframe[target]
        for feature in dataframe[feature])
    return cfd

cfd = nltk.ConditionalFreqDist((genre, word) for genre in brown.categories() for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
print(cfd.tabulate(conditions=genres, samples=modals))
### Models

def logistic_regression(dataframe:pd.DataFrame,feature:str,target:str):
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