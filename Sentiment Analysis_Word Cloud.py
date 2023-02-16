import pandas as pd #library for handling, loading, and manipulating data
import nltk #platform for building Python programs that work with human language data. Contains text processing libraries
import re #module provides regular expression support
import emoji #module for emojis
import string #for string operators
from nltk.stem import PorterStemmer #module for stemming
from nltk.tokenize import TweetTokenizer #module for tokenizing strings
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from nltk.corpus import stopwords # Import the english stop words list from NLTK
nltk.download('stopwords')
stopwords_english = stopwords.words('english')



# Import the training dataset Sentiment140 corpus into a Pandas dataframe
df = pd.read_csv("training_1600000.csv", encoding = 'latin', header=None)
# Specify the name for each column in the Sentiment140 dataset
df.columns = ['sentiment', 'ID', 'date', 'query', 'name', 'text']
# Since positive tweets are labeled 4 in the Sentiment140 dataset, we replaced 4 with 1 to sum up positive tweets more easily.
df['sentiment'] = df['sentiment'].replace(4,1)

# Import datasets for three periods.
# Import the period 1 tweet dataset
dataset1 = pd.read_csv("Jan_2020_p1.csv")
# Import the period 2 tweet dataset
dataset2 = pd.read_csv("Feb_2020_p2.csv")
# Import the period 3 tweet dataset
dataset3a = pd.read_csv("Mar_2020_1_p3.csv")
dataset3b = pd.read_csv("Mar_2020_2_p3.csv")
dataset3c = pd.read_csv("Mar_2020_3_p3.csv")
dataset3d = pd.read_csv("Mar_2020_4_p3.csv")



#Create a function to clean the tweets
def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # Removed @mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # Removing #hashtags
    text = re.sub(r'https?:\/\/\S+', '', text) # Remove the hyperlink
    text = text.lower() #lowerccase
    #turn emojis into corresponding text
    text = emoji.demojize(text) 
    text = text.replace("_"," ")
    text = ' '.join(text.split())
    
    return text

# Instantiate tokenizer class.
# First parameter: don't downcase text. Second parameter: remove Twitter handles of text.
# Third parameter: replace repeated character sequences of length 3 or greater.
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)

# Split each tweet into tokens/words
def tokenize_tweet(tweet):
    
    tweet_tokens = tokenizer.tokenize(tweet)
    
    return tweet_tokens

# Return all sets of punctuation
punctuations = string.punctuation

# Remove stopwords and punctuations
def remove_stopwords_punctuations(tweet_tokens):
    
    tweets_clean = []
    
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in punctuations):
            tweets_clean.append(word)
            
    return tweets_clean

# Transform word into its stem by removing suffixes such as -ed, -ing, and -ion
stemmer = PorterStemmer()

def get_stem(tweets_clean):
    
    tweets_stem = []
    
    for word in tweets_clean:
        stem_word = stemmer.stem(word)
        tweets_stem.append(stem_word)
        
    return tweets_stem

#Combining all preprocess techniques
def process_tweet(tweet):
    
    processed_tweet = cleanTxt(tweet)
    tweet_tokens = tokenize_tweet(processed_tweet)
    tweets_clean = remove_stopwords_punctuations(tweet_tokens)
    tweets_stem = get_stem(tweets_clean)
    
    return tweets_stem



# Convert a collection of training text dataset to a matrix of token counts.
vectorizers = CountVectorizer(analyzer = process_tweet, dtype = 'uint8')
df_countvectorizer = vectorizers.fit_transform(df['text'])

# The Sentiment140 dataset is split into training and test data with split ratio of 75:25. 
X_train, X_test, y_train, y_test = train_test_split(df_countvectorizer, df['sentiment'],test_size = 0.25, random_state = 0)

# Fetch the dimensions of the X_train, X_test, y_train, y_test dataset
X_train.shape
X_test.shape
y_train.shape
y_test.shape

# Ignore all warnings from Python
import warnings
warnings.filterwarnings('ignore')

# Import the MultinomialNB classification
from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
#Fit the classifier into the training data
NB_classifier.fit(X_train, y_train)
# Crete the predicted tags of 0 or 1 for sentiment 
from sklearn.metrics import classification_report
y_predict_test = NB_classifier.predict(X_test)
#Extract the accuracy score from the Multinomial NB classifier
NB_acc = accuracy_score(y_predict_test, y_test)
print("Test accuracy: {:.2f}%".format(NB_acc*100))

# Import the BernoulliNB classification
from sklearn.naive_bayes import BernoulliNB
BNBmodel = BernoulliNB(alpha = 2)
#Fit the classifier into the training data
BNBmodel.fit(X_train, y_train)
# Crete the predicted tags of 0 or 1 for sentiment 
y_predict_test1 = BNBmodel.predict(X_test)
#Extract the accuracy score from the BernoulliNB classifier
BNB_acc = accuracy_score(y_predict_test1, y_test)
print("Test accuracy: {:.2f}%".format(BNB_acc*100))

# Import the Logistic Regression classification
logreg = LogisticRegression()
# Fit the classifier into the training data
logreg.fit(X_train,y_train)
# Crete the predicted tags of 0 or 1 for sentiment 
logreg_pred = logreg.predict(X_test)
# Extract the accuracy score from the BernoulliNB classifier
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))

# Import GridSearch CV module
from sklearn.model_selection import GridSearchCV
# Setup the hyperparameter grid for Logistic Regression classification
param_grid={'C':[0.001, 0.01, 0.1, 1, 10]}
# Instantiate the GridSearchCV object: grid
grid = GridSearchCV(LogisticRegression(), param_grid)
# Fit it to the data
grid.fit(X_train,y_train)
# Print the tuned parameters
print("Best parameters:", grid.best_params_)
# Extract the accuracy score from the tuned Logistic Regression classifier
y_pred = grid.predict(X_test)
logreg1_acc = accuracy_score(y_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg1_acc*100))

# Import the LinearSVC classification
from sklearn.svm import LinearSVC
# Train the model on training dataset
SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)
# Extract the accuracy score from the SVCmodel classifier
svc_pred = SVCmodel.predict(X_test)
svc_acc = accuracy_score(svc_pred, y_test)
print("Test accuracy: {:.2f}%".format(svc_acc*100))

# Setup the hyperparameter grid for LinearSVC classification
param_grid = {'C': [0.01, 0.1, 1, 10], 'gamma':[0.01,1]}
grid = GridSearchCV(SVCmodel, param_grid, refit = True)
# Fit the model for grid search
grid.fit(X_train, y_train)
# Print best parameter after tuning
print("Best parameters:", grid.best_params_)
# Print how the model looks after hyper-parameter tuning
print(grid.best_estimator_)
# Extract the accuracy score from the tuned LinearSVC classifier
y_pred4 = grid.predict(X_test)
logreg_acc4 = accuracy_score(y_pred4, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc4*100))

# Convert a collection of tweets from period 1 into a matrix of token counts.
dataset1_test = vectorizers.transform(dataset1['Text'])
# Fetch the dimensions of the period 1 tweet dataset
dataset1_test.shape
# Crete the predicted tags of 0 or 1 for sentiment using Logistic Regression classification
y_dataset1 = grid.predict(dataset1_test)
# Add a 'Sentiment' column in the period 3 dataset with predicted tags
dataset1['Sentiment'] = y_dataset1
# Export and save the dataset into a csv file
dataset1.to_csv(r'C:\Users\nguye\Twitter\Jan_2020_1_p1_new.csv', index=False)

dataset2_test = vectorizers.transform(dataset2['Text'])
dataset2_test.shape
y_dataset2 = grid.predict(dataset2_test)
dataset2['Sentiment'] = y_dataset2
dataset2.to_csv(r'C:\Users\nguye\Twitter\Feb_2020_p2_new.csv', index=False)

dataset3a_test = vectorizers.transform(dataset3a['Text'])
dataset3a_test.shape
y_dataset3a = grid.predict(dataset3a_test)
dataset3a['Sentiment'] = y_dataset3a
dataset3a.to_csv(r'C:\Users\nguye\Twitter\Mar_2020_1_p3_new.csv', index=False)

dataset3b_test = vectorizers.transform(dataset3b['Text'])
dataset3b_test.shape
y_dataset3b = grid.predict(dataset3b_test)
dataset3b['Sentiment'] = y_dataset3b
dataset3b.to_csv(r'C:\Users\nguye\Twitter\Mar_2020_2_p3_new.csv', index=False)

dataset3c_test = vectorizers.transform(dataset3c['Text'].values.astype('U'))
dataset3c_test.shape
y_dataset3c = grid.predict(dataset3c_test)
dataset3c['Sentiment'] = y_dataset3c
dataset3c.to_csv(r'C:\Users\nguye\Twitter\Mar_2020_3_p3_new.csv', index=False)

dataset3d_test = vectorizers.transform(dataset3d['Text'].values.astype('U'))
dataset3d_test.shape
y_dataset3d = grid.predict(dataset3d_test)
dataset3d['Sentiment'] = y_dataset3d
dataset3d.to_csv(r'C:\Users\nguye\Twitter\Mar_2020_3_p4_new.csv', index=False)

################################################################################
#Word Cloud

# Import WordCloud module 
from wordcloud import WordCloud
# Import module that provide numerous tools to deal with filenames, paths, directories.
import sys, os
import glob
# Change the current working directory to the first item of the list "path"
os.chdir(sys.path[0])

# Specify the path
path = "/Users/nguye/Twitter/March 2020_p3"
# Find all CSV files in the specified directory path and stored in file_list
file_list = glob.glob(path + "/*.csv")
print('File names:', file_list)

# Concatenate 4 tweet datasets of period 3 (dataset3a, dataset3b, dataset3c, dataset3d) into 1 large dataset3
df_list = (pd.read_csv(file, low_memory=False) for file in file_list)
dataset3 = pd.concat(df_list, ignore_index=True)

# Convert the dataframes of dataset1, dataset2, dataset3 into a list of tweets
text1 = dataset1.Text.values.tolist()
text2 = dataset2.Text.values.tolist()
text3 = dataset3.Text.values.tolist()

# Speficy words addressing COVID-19 and filler words to be removed
removed_words = ["coronavirus", "corona virus", "coronaviru", "coronavirusoutbreak", "ncov", "coronaviruswho", "wuhancoronavirus", "covid-19", "covid19", "covid", "_19", "u", "see", "say", "think", "19", "said", "say", "think", "look", "hey", "know", "saying", "looking", "thought", "thinking", "making", "many", "much", "via", "want", "another", "few", "thing", "telling", "next week", "last week", "seem", "something", "anything", "use", "using", "talking", "ye", "instead", "lot", "one", "even", "still", "got", "going", "made", "make", "especially", "specially", "actually", "etc", "put", "used"]

# Combine all preprocess techniques and process input list of tweets
# Create an empty list to store processed tweets.
final_tweet2 = []

for processing_text in text2:
    processing_text = str(processing_text)
    cleaned_text = cleanTxt(processing_text)
    # Remove words that are listed in removed_words
    removed_tweet = ' '.join(i for i in cleaned_text.split() if i not in removed_words)
    tweet_tokens = tokenize_tweet(removed_tweet)
    tweets_clean = remove_stopwords_punctuations(tweet_tokens)

    # Add processed tweet into final_tweet2 list.
    final_tweet2.append(tweets_clean)

# Join all processed tweets into a listToStr2 string
listToStr2 = ' '.join(final_tweet2)

# Initiate the WordCloud module
wc = WordCloud(
        background_color = "white",
        height = 400,
        width = 600
)

# Create a word cloud for processed tweets
wc.generate(listToStr2)

# Save the word cloud as a png file
wc.to_file('period1.png')


