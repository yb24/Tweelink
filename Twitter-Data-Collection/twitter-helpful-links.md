Blog (Tutorial for automation and json to csv):
	https://towardsdatascience.com/an-extensive-guide-to-collecting-tweets-from-twitter-api-v2-for-academic-research-using-python-3-518fcb71df2a

Automation:
	https://developer.twitter.com/en/docs/twitter-api/pagination

Topic discovery with twitter data:
	https://blog.twitter.com/developer/en_us/topics/tips/2021/three-approaches-to-topic-discovery-with-twitter-data

Postman:
	https://developer.twitter.com/en/docs/tutorials/postman-getting-started

Twurl:
	https://developer.twitter.com/en/docs/tutorials/using-twurl

Tweet Types:
	https://developer.twitter.com/en/docs/tutorials/determining-tweet-types

Stream 1% of all Tweets in real time:
	https://developer.twitter.com/en/docs/twitter-api/tweets/volume-streams/api-reference/get-tweets-sample-stream#Optional

Sentiment Analysis:
	https://developer.twitter.com/en/docs/tutorials/how-to-analyze-the-sentiment-of-your-own-tweets

Location:
	https://developer.twitter.com/en/docs/tutorials/filtering-tweets-by-location

Operators by product:
	https://developer.twitter.com/en/docs/twitter-api/enterprise/rules-and-filtering/operators-by-product

Search recent:
	https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-recent

Twitter API v2:
	sort_order:
		https://twittercommunity.com/t/introducing-the-sort-order-parameter-for-search-endpoints-in-the-twitter-api-v2/166377
	API v2 sample code:
		https://github.com/twitterdev/Twitter-API-v2-sample-code

Query building tips:
	https://developer.twitter.com/en/docs/tutorials/building-high-quality-filters
	https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query

Note:
	
	1. context annotations are delivered as a context_annotations field in the payload, and are semantically classified based on the Tweet text and result in domain and entity pairings

	2. context type, context name

	3. Tweet Location Data: Tweet Location, Account Location

	4. Important note: Retweets can not have a Place attached to them, so if you use an operator such as has:geo, you will not match any Retweets. has:geo is not available in Sandbox mode. Try -is:retweet

	5. Note that the “coordinates” attributes is formatted as [LONGITUDE, latitude], while the “geo” attribute is formatted as [latitude, LONGITUDE].

How to collect twitter dataset:

	1. for any given topic, set start_time and end_time differing by 1 day for 7 days for 1 month.

	2. every person will run their list of a particular query for 7 days.

	3. keep in mind the path of csv file (change if needed for your system). DO NOT track/commit this csv file (see .gitignore for this).