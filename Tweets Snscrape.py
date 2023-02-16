import snscrape.modules.twitter as sntwitter
from time import sleep
import time
import pandas as pd 

# Set the maximum number of tweets to receive
maxTweets = 150000

# Creating a list to append tweet data 
tweets_list = []
source="Twitter"

# Filter and collect tweets with the following hashtags 
keyword = "#coronavirus OR #CoronavirusOutbreak OR #pandemic OR #CoronavirusPandemic OR #WuhanCoronavirus"

# There are two ways to specify the timeframe for collecting tweets. One is Unix timestamp and the other is date.
# For example, 'since_time:1584532841 until_time:1584576000' and "since:2020-03-13 until:2020-03-14"
# Using TwitterSearchScraper to scrape data and append tweets to the list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(keyword + "since_time:1584532841 until_time:1584532942").get_items()):
    if i>maxTweets:
        break
    print(i)
    tweets_list.append([tweet.id,tweet.user.username,tweet.content,tweet.date,tweet.retweetCount,tweet.likeCount, tweet.lang])

# Creating a Pandas dataframe from the tweets list above
tweets_df = pd.DataFrame(tweets_list,columns=['Tweet_ID', "Account_Name", 'Text', 'Datetime','Number_Retweets', 'Number_Likes', "Lang"])

# Filtering and collecting only English tweets
tweets_df = tweets_df[(tweets_df['Lang']=='en')]

# Export the Pandas dataframe into a csv file
tweets_df.to_csv(r'C:\Users\nguye\Twitter\Mar_2020_p1.csv', index=False)

