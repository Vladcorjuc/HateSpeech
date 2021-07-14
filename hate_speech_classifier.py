import csv
# for the word frequency function
from nltk.tokenize import TweetTokenizer
from gensim.corpora import Dictionary
import itertools
from collections import defaultdict
import pandas as pd

# for the wordcloud function
from wordcloud import WordCloud
import matplotlib.pyplot as plt


#vectorization
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#models
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
#nltk.download('wordnet')


processed_data = pd.read_csv('data\\processed\\data.csv')
tknz = TweetTokenizer()
stemmer = SnowballStemmer("english")

def clean_dictionary(dictionary):
	dictionary = dictionary.str.replace("n't", "not")
	dictionary = dictionary.str.replace("'ve ", "")
	dictionary = dictionary.str.replace("'re", "")
	dictionary = dictionary.str.replace("wan na", "wanna")
	dictionary = dictionary.str.replace("gon na", "gonna")
	dictionary = dictionary.str.replace(" '", "")
	dictionary = dictionary.str.replace("' ", "")
	return dictionary


def tokenize_tweet(s):
    tokens = tknz.tokenize(s)
    return [w for w in tokens]

def get_tokens_frequency_df(series):
    corpus_lists = [doc for doc in series.dropna() if doc]
    dictionary = Dictionary(corpus_lists)
    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus_lists]
    token_freq_bow = defaultdict(int)
    for token_id, token_sum in itertools.chain.from_iterable(corpus_bow):
        token_freq_bow[token_id] += token_sum

    return pd.DataFrame(list(token_freq_bow.items()), columns=['token_id', 'token_count']).assign(
        token=lambda df1: df1.apply(lambda df2: dictionary.get(df2.token_id), axis=1),
        doc_appeared=lambda df1: df1.apply(lambda df2: dictionary.dfs[df2.token_id], axis=1)).reindex(
        labels=['token_id', 'token', 'token_count', 'doc_appeared'], axis=1).set_index('token_id')


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

def plot_word_cloud(dictionary, top_n):
    word_cloud = WordCloud(background_color='white', colormap='magma', contour_width=1,
                           contour_color='orange', relative_scaling=0.5)

    sorted_freq_dict = dict(dictionary[['token', 'token_count']].nlargest(top_n, columns='token_count').values)
    wc = word_cloud.generate_from_frequencies(frequencies=sorted_freq_dict, max_font_size=40)

    _, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Term Frequency', fontsize=16)

    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.show()


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

tfidf_vectors = TfidfVectorizer(max_df=0.90, min_df=2, max_features=9000,
	stop_words='english',
	ngram_range=(1, 3))
def vectorize(dictionary):
	tfidf_only_fit = tfidf_vectors.fit(dictionary['tweet_without_stopwords_and_2'])

	pickle_out_only_fit = open('tfidf_pickle_fit', 'wb')
	pickle.dump(tfidf_only_fit, pickle_out_only_fit)

	tfidf = tfidf_vectors.fit_transform(dictionary['tweet_without_stopwords_and_2'])
	df_vector = pd.DataFrame(tfidf.todense(),columns = tfidf_vectors.get_feature_names())
	return df_vector


#negative_tweet = processed_data['tweet_without_stopwords_and_2'].loc[processed_data['neg_label']==1]
#positive_tweet = processed_data['tweet_without_stopwords_and_2'].loc[processed_data['neg_label']==0]

#negative_tweet = clean_dictionary(negative_tweet)
#positive_tweet = clean_dictionary(positive_tweet)

#negative_tweet_tokenized = negative_tweet.apply(tokenize_tweet)
#positive_tweet_tokenized = positive_tweet.apply(tokenize_tweet)

#negative_tweet_freq_tokens = get_tokens_frequency_df(negative_tweet_tokenized)
#positive_tweet_freq_tokens = get_tokens_frequency_df(positive_tweet_tokenized)

#plot_word_cloud(negative_tweet_freq_tokens,100)
#plot_word_cloud(positive_tweet_freq_tokens,100)


df_vector = vectorize(processed_data)
target = processed_data['neg_label']
x_train, x_test, y_train , y_test = train_test_split(df_vector, target, 
                                                     test_size =.2, random_state=101 )
X_train, x_val, Y_train , y_val = train_test_split(x_train,y_train, 
                                                     test_size =.2, random_state=101 )


logmod = LogisticRegression(random_state=1002,verbose=1)
logmod.fit(x_train, y_train)
predict = logmod.predict(x_test)
print(accuracy_score(y_test,predict))


