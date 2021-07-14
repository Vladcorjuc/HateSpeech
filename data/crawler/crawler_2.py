from twitter import *
import folium
import pandas as pd
import csv
import argparse
import sys
from PIL import Image
import requests
import io
import base64
import os


#Get the api object using Oauth authentification with my Twitter Api Dev Account
api=Twitter(auth = OAuth('1334043943701450753-R5TJpyUJcKARRO5Ne7BVKyjjUGu5cN',
	'jthsdSRDENBhe4JgahtycDp7tORvtzv5ieHNF2X5kA3QN',
	'rODOfZSSTwkvALJUQU5n1TkIa',
	'3wQWRW3KFhJUNnawrgInuoLKK5f4fnPgMLt2Snbew5qudgsBWj'))


def compute_mean(coord_box):
	"""
		Computes mean of four geo-coordinates

		params:
			coord_box([[x,y],[x,y],[x,y],[x,y]]) : polygon coords area
		returns:
			longitude(float),latitude(float) : mean longitude and latitude of the four coords
	"""
	coord_sum=[a + b + c + d for a,b,c,d in zip(coord_box[0][0],coord_box[0][1],coord_box[0][2],coord_box[0][3])]
	coord_sum=[coord/4 for coord in coord_sum]
	return coord_sum[1],coord_sum[0]

def get_arguments(argv):
	"""
		Set and parse arguments from the command line

		params:
			argv : string of command line arguments received
		returns:
			hastag(string), number_tweets(int) : a string representing the hashtag and the number of tweets that need to be displayed
	"""
	parser = argparse.ArgumentParser(description='Search tweets by hastag and create a map (html_file) with the tweets',epilog="Made by VLAD CORJUC")
	parser.add_argument('--hashtag','-#',type=ascii,
                    help='The hastag to be searched',required=True)
	parser.add_argument('-n','--number-tweets',nargs="?",const=5,default=5,type=int,help='How many tweets to search(default 5)')
	
	args = vars(parser.parse_args(argv[1:]))

	hashtag=args["hashtag"].strip("\'")
	number_tweets= args['number_tweets']
	return hashtag,number_tweets

def retrive_tweets():
	"""
		Retrive tweets that have geo-positional information.

		params:
			hashtag(string) : hashtag to be searched
			number_tweets(int) : number of tweets that need to be retrived
		returns:
			tweets(array of dicts) : array o dictionaries containing info about the tweet

	"""

	tweets=[]
	number_tweets=10
	with open("hate_speech_words.txt") as hashtags:
		try:
			for hashtag in hashtags:
				returned_results=0
				last_id=None
				hashtag=hashtag.strip()
				print(hashtag,"-START")
			#To obtain the desired number of tweets we need to iterate multiple times because we have tweets with 0 geo-positional information
			#	and the maximum search count accepted by the api is 100
				while returned_results < number_tweets:
					print(returned_results)
					#Get new tweets until hitting last_id or the max_count
					results = api.search.tweets(
				    q="{0}".format(hashtag),count=100,max_id = last_id)

					for result in results['statuses']:
						#save only those tweet that contains geographical coordinates
						if result["place"] and returned_results<number_tweets:
							returned_results+=1
							bounding_box=result["place"]["bounding_box"]
							coordinates=bounding_box["coordinates"]
							
							mean_latitude,mean_longitude=compute_mean(coordinates)

							tweet_text=result["text"]
							tweet_user = result["user"]
							tweet_date=result["created_at"]
							tweet_media=None
							tweet_hashtags=None
							if result["entities"]:
								if "media" in result["entities"]:
									tweet_media=result["entities"]["media"]
								if "hashtags" in result["entities"]:
									tweet_hashtags=result["entities"]["hashtags"]

							tweets+=[{"latitude":mean_latitude,"longitude":mean_longitude,"text":tweet_text,
							"user":tweet_user,"date":tweet_date,"media":tweet_media,"hashtags":tweet_hashtags}]			
							
						last_id = result["id"]
				print(hashtag,"-DONE")
		except:
			return tweets
	return tweets

def create_map(tweets_info):
	"""
		Creates a Folium Map object containing markers at the tweet coordinates

		params:
			tweet_info (pandas csv file) : csv containing different tweets with their info
		return:
			world_map (folium.Map Object) : map containing the markers
	"""
	center = [0., 0.]
	world_map = folium.Map(location=center, zoom_start=3)
	print(tweets_info.iterrows)
	for index, tweet_info in tweets_info.iterrows():
	    location = [tweet_info['latitude'], tweet_info['longitude']]
	    tweet_user=eval(tweet_info["user"])

	    #Retrive profile picture of the user from url
	    profile_img_url=tweet_user["profile_image_url"]
	    response = requests.get(profile_img_url)
	    profile_img = io.BytesIO(response.content)
	    encoded_image = base64.b64encode(profile_img.read())

	    #Html that will be displayed by the marker popup, it contains the username, the profile picture,the tweet text and the date when it was created
	    html = '<body style="background-color:#E1E8ED"><img src="data:image/png;base64,{img}">  <b>{username}</b><br><p>{text}<br><small>Tweeted At: {date}</small></p>'.format(
	    	username=tweet_user['screen_name'],
	    	img=encoded_image.decode('UTF-8'),
	    	text=tweet_info['text'],
	    	date=tweet_info['date']
	    	)
	    #Frame containing the html to be displayed by the popups
	    iframe = folium.IFrame(html, width=300, height=200)
	    popup = folium.Popup(iframe, max_width=2650)
	    #Add Marker at the location using the previous created popup frame
	    folium.Marker(location, popup = popup).add_to(world_map)

	return world_map

def main(argv):

	#hashtag,number_tweets = get_arguments(argv)

	tweets=retrive_tweets()

	try:
	    with open("tweets-info.csv", 'a',encoding="utf-8") as csvfile:
	        writer = csv.DictWriter(csvfile, fieldnames=tweets[0].keys())
	        writer.writeheader()
	        writer.writerows(tweets)
	        print("SAVED tweets as: tweets-info.csv")
	except IOError:
	    print("I/O error when handling the csv file!")


	#tweets_info = pd.read_csv('tweets-info.csv', encoding='utf-8')
	#world_map=create_map(tweets_info)
	# save map to html file and open it in browser
	#try:
		#world_map.save('map.html')
		#print("SAVED tweets map as: map.html")
		#os.system("start map.html")
		#print("OPEN map.html")
	#except Exception as e:
		#raise e

if __name__ == '__main__':
	main(sys.argv)