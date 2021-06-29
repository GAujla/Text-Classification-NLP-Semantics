from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import time


#Import and read the dataset
df = pd.read_csv(r'data/Corona_NLP_train.csv')

# Monitor the time
start = time.time()
# Possible sentiments a tweet may have
df_sentiment = df.Sentiment.unique()


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.tolist.html
# Gets the values of how many times each sentiment value appeared and puts in
# order of list
secondCom = df[r'Sentiment'].value_counts().index.tolist()
print('The possible Sentiments a tweet has are:',secondCom)
print(" ")
# Prints second value of most common sentiment
print('The second most common sentiment in tweets is:',secondCom[1])
print("")
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
# https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html
# Keeps all Extremely Positive tweets
df1 = df[df['Sentiment'].str.contains('^Extremely Positive', na=False)]
# Groups the Sentiment and TweetAt together
# Counts how frequently they appear
new = df1.groupby(['Sentiment','TweetAt']).size().reset_index(name='count')
# Sort the values
positive = (new.sort_index(axis=0).sort_values(['count'], ascending=False))
print("Display the amount of Extremely poisitive tweets and the date of which"
      " there was the highest number of them")
print(positive.head(1))
print(" ")


# https://stackoverflow.com/questions/22588316/pandas-applying-regex-to-replace-values
# https://docs.python.org/3/howto/regex.html
# Converts messages of tweets to lower case
# Replaces all non-alphabetic lettering with whitespaces
df['CharChange']= df["OriginalTweet"].str.lower().replace(r'[^a-z]+',' ', regex=True)
print('Leterring is now lower case and all non-alphabetic letters are converted '
      'into white space:')
print(df[r'CharChange'])
print(" ")

# https://www.programcreek.com/python/example/106181/sklearn.feature_extraction.stop_words.ENGLISH_STOP_WORDS
# Words are tokenized and counted
stop_words = set(sklearn.feature_extraction.text.ENGLISH_STOP_WORDS)
print('Total number of words including repetitions:',
      df['CharChange'].str.split().str.len().sum())
print(" ")

# Total unique words are calculated
word_amount = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
X_train_counts = word_amount.fit_transform(df[r'CharChange'])
print('Number of Distinct words:',X_train_counts.shape[1])
print("")

# Display the top 10 most frequent words used in the tweets
test = df['CharChange'].str.split(expand=True).stack().value_counts()
print('Top 10 Most frequent words are:')
print(test.T.sort_values(ascending=False).head(10))
print("")
# https://stackoverflow.com/questions/52421610/how-to-replace-all-words-of-length-less-than-3-in-a-string-with-space-in-j
# https://docs.python.org/3/library/re.html

# StopWords and Words with less than two characters are removed.
less_than_two = re.compile(r'\b\w{,2}\b')
df['new'] = df['CharChange'].apply(lambda s: less_than_two.sub('',s))
escaped = map(re.escape, stop_words)
pattern = re.compile(r'\b(' + r'|'.join(escaped) + r')\b')
df['removal'] = df['new'].apply(lambda s: pattern.sub('',s))
# Total number of words after removal are recalculated
print('Number of words (after removal):',df['removal'].str.split()
      .str.len().sum())
print("")
# Top 10 words are recalculated and displayed again
remove_stop1 = df['removal'].str.split(expand=True).stack().value_counts()
print("After removal of words ")
print(remove_stop1.T.sort_values(ascending=False).head(10))
print("")

# Multinomial Naive Bayes classifier is implemented for the
# Coronavirus NLP semantics to predict the classification of tweets.
# Corpus is stored in a numpy array
# Trained on training set
vectorizer = CountVectorizer()
clf = MultinomialNB()
X = vectorizer.fit_transform(df['removal'])
y = df['Sentiment']
clf.fit(X, y)
x = clf.predict(X)
acc = accuracy_score(x, y)
error_rate = 1 - acc
print('The error rate is',error_rate)
print("")
# Display the frequency of words in histogram which will be shown in the outputs
# folder. Log is used to improve the readability.
# Plots word frequencies using log to improve readability

new = pd.DataFrame(df['removal'].str.split().apply(set).tolist()).stack().\
    value_counts(ascending = True)
plt.yscale("log")
plt.plot(range(len(new)),new)
plt.savefig('outputs/word frequency.jpg')
plt.show()
print('The total time taken',time.time()-start)









