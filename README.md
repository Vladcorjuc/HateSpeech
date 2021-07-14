# Hate Speech - Twitter
## twitter hate speech detection and similarities between positive detected tweets
Wordcloud of bad words:<br> <img src="https://github.com/Vladcorjuc/HateSpeech/blob/main/wordclouds/negative_tweets_wordcloud.png" alt="Wordcloud of bad words" width="400"/><br>
Wordcloud of good words: <br> <img src="https://github.com/Vladcorjuc/HateSpeech/blob/main/wordclouds/positive_tweets_wordcloud.png" alt="Wordcloud of good words" width="400"/><br>
Graph:
<br> <img src="https://github.com/Vladcorjuc/HateSpeech/blob/main/graph/Graph.png" alt="Graph" width="500"/><br>
#### Nodes: tweet positive classified by the detection algorithm; the size of a node is given by its influence (followers number)<br>
#### Edges: Between tweets that present similar ideas (similar topic), coloring according to the 50 topics created. The weight of the edge is directly proportional to the degree of similarity.
