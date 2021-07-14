from hate_speech_classifier import *

def construct_nodes():
	_processed_data = pd.read_csv('data\\to_classify\\tweets.csv',encoding = "ISO-8859-1")


	ids = _processed_data["id"]
	corpus = _processed_data["text"]

	processed_text=[preprocess(corpus_element) for corpus_element in corpus]

	dictionary = gensim.corpora.Dictionary(processed_text)
	bow_corpus = [dictionary.doc2bow(doc) for doc in processed_text]

	lda_model =  gensim.models.LdaMulticore(bow_corpus, 
	                                   num_topics = 50, 
	                                   id2word = dictionary,
	                                   workers = 1)

	for idx, topic in lda_model.print_topics(-1):
	    print("Topic: {} \nWords: {}".format(idx, topic ))
	    print("\n")

	columns_name=['id','date','screen_name','text','username','background_img_url','profile_image','followers_count','friends_count',
	                'retweet_count','favorite_count','geo','coordinates','hashtags','type','topic','topic_score']
	with open("graph\\nodes.csv", 'a',encoding="ISO-8859-1") as csv_write_file:
		writer = csv.DictWriter(csv_write_file, fieldnames=columns_name,)
		writer.writeheader()

		for index,tweet in _processed_data.iterrows():
			print("We are at tweet -> ",index)		
			text=tweet['text']
			tweet_id = tweet["id"]
			date=tweet["date"]

			retweet_count=tweet['retweet_count']
			favorite_count=tweet['favorite_count']

			username=tweet['username']
			screen_name=tweet['screen_name']

			followers_count=tweet['followers_count']
			friends_count=tweet['friends_count']
			profile_image=tweet['profile_image']
			background_img_url = tweet['background_img_url']

			hashtags=tweet["hashtags"]

			phrase=np.array([str(text)])
			tfidf = tfidf_vectors.transform([phrase[0]])
			phrase_vector = pd.DataFrame(tfidf.todense(),columns = tfidf_vectors.get_feature_names())

			_type =logmod.predict(phrase_vector)

			bow_vector = dictionary.doc2bow(preprocess(text))               
			index, score  = sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1])[0]

			topic = index
			topic_score = score


			tweets={'id':tweet_id,'date':date,'screen_name':screen_name,'text':text,'username':username,
			'background_img_url':background_img_url,"profile_image":profile_image,'followers_count':followers_count,
			'friends_count':friends_count,'retweet_count':retweet_count,'favorite_count':favorite_count,"hashtags":hashtags,'type': _type,
			'topic':topic,'topic_score':topic_score}

			writer.writerow(tweets)
	print("->>DONE")
		
def construct_edges():
	_processed_data = pd.read_csv("nodes.csv",encoding = "ISO-8859-1")

	columns_name=['Source', 'Target','Label','Weight']

	with open("graph\\edges.csv", 'a',encoding="ISO-8859-1") as csv_write_file:
		writer = csv.DictWriter(csv_write_file, fieldnames=columns_name,)
		writer.writeheader()

		for i in range(len(_processed_data)-1):
			for j in range(i+1,len(_processed_data)):
				tweet1 = _processed_data.iloc[i]
				tweet2 = _processed_data.iloc[j]

				if(int(tweet1['topic']) != int(tweet2['topic'])):
					continue

				print("Create edge (",i,",",j,")")
				edge = {"Source":tweet1['id'],"Target":tweet2['id'],"Label":tweet1['topic'],"Weight": 1.0 - abs(float(tweet1['topic_score'])-float(tweet2['topic_score']))}
				writer.writerow(edge)
				print("DONE")
	print("->>DONE")


def main():
	construct_nodes()
	construct_edges()

if __name__ == '__main__':
	main()