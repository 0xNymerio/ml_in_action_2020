import csv
import itertools as it

with open('./Naive_Bayes/tweets/100_tweets_positive.csv') as f:
    reader=csv.reader(f)
    for i in range(1,101):
            tweet = next(it.islice(reader, i))
            with open('./Naive_Bayes/tweets/tweets_positive/'+str(i)+'.txt',"w") as output:
                output.write(str(tweet))
