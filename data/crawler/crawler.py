from TwitterSearch import *
import csv
import time
import datetime


hate_speech_words=[]
with open("hate_speech_words.txt") as hate_words:
    for word in hate_words:
        hate_speech_words+=[word.split()[0]]
try:
    ts = TwitterSearch(
        consumer_key = 'rODOfZSSTwkvALJUQU5n1TkIa',
        consumer_secret = '3wQWRW3KFhJUNnawrgInuoLKK5f4fnPgMLt2Snbew5qudgsBWj',
        access_token = '1334043943701450753-R5TJpyUJcKARRO5Ne7BVKyjjUGu5cN',
        access_token_secret = 'jthsdSRDENBhe4JgahtycDp7tORvtzv5ieHNF2X5kA3QN'
     )
    columns_name=['id','date','screen_name','text','username','background_img_url','profile_image','followers_count','friends_count',
                    'retweet_count','favorite_count','geo','coordinates','hashtags','type']

    with open("crawled_tweets.csv", 'a') as csv_write_file:
        writer = csv.DictWriter(csv_write_file, fieldnames=columns_name)
        #writer.writeheader()

        i=0
        for word in hate_speech_words:
            print(word)
            #if i>0 : 
                #time.sleep(60*15)
            i+=1
            for i in range(1):
                print(i)
                tso = TwitterSearchOrder()
                tso.set_negative_attitude_filter()
                tso.set_keywords([word])
                tso.set_language('en')

                try:
                    response=ts.search_tweets(tso)
                except Exception as e:
                    print(e)
                    break
                next_max_id=0
                sleep_for=60
                last_amount_of_queries=0
                if(len(response['content']['statuses'])==0):
                    continue
                for tweet in response['content']['statuses']:
                    current_amount_of_queries = ts.get_statistics()[0]
                    if not last_amount_of_queries == current_amount_of_queries:
                        last_amount_of_queries = current_amount_of_queries
                        #time.sleep(sleep_for)
                    try:
                        text=tweet['text'].encode('cp850', errors='replace')
                        tweet_id = tweet['id']
                        date=tweet["created_at"]
                        retweet_count=tweet['retweet_count']
                        favorite_count=tweet['favorite_count']
                        geo=tweet['geo']
                        coordinates=tweet['coordinates']

                        user=tweet['user']
                        username=user['name']
                        screen_name=user['screen_name']
                        location=user['location']
                        description=user['description']

                        user_entities=user['entities']
                        followers_count=user['followers_count']
                        friends_count=user['friends_count']
                        background_img_url=user['profile_background_image_url']
                        profile_image=user['profile_image_url']

                        hashtags=None
                        if 'entities' in tweet:
                            hashtags=tweet['entities']['hashtags']

                        tweets={'id':tweet_id,'date':date,'screen_name':screen_name,'text':text,'username':username,
                        'background_img_url':background_img_url,"profile_image":profile_image,'followers_count':followers_count,
                        'friends_count':friends_count,'retweet_count':retweet_count,'favorite_count':favorite_count,'geo':geo,
                        'coordinates':coordinates,"hashtags":hashtags,'type':None}

                        writer.writerow(tweets)

                    except Exception as e:
                        #print(e)
                        pass

                    if (tweet_id < next_max_id) or (next_max_id == 0):
                        next_max_id = tweet_id
                        next_max_id -= 1
                tso.set_max_id(next_max_id)
except TwitterSearchException as e:
    print(e)