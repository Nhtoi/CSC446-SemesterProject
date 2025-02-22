CHALLENGES:
 [] HOW TO EXTRACT INFORMATION FROM WEBSITE
 [] CLEAN THE DATA, NO URL’S, SPECIAL CHARACTERS ETC
 [] TRAIN AI (BERTSUM AND PYTORCH)
 [] BUILD FRONT-END (FOR EXTRA POINTS/WOW-FACTOR I ASSUME)


HOW TO EXTRACT INFORMATION FROM WEBSITES	

Used Reddit API and praw to scrape posts, and gather comments along with infromation such as upvotes. Cannot track downvotes.
but upvotes alone should be a decent indication that a comment is a good addition to the thread and not a troll comments, 
however comments that are found funny could be added to this meaning that our model may think that a comment with a high ammount
of upvotes is a helpful comment always, could implement some sort of filter to combat this however, I don't think it will be necessary
extracting comments from inside comments has not been added yet, and im still not quite sure if it is necessary. Unless we need more data
For our training we need to extract upvotes, comments, anything really that can indicate a good comment and helpful for our summarization.

Could be a google chrome plugin, that can communicate with a python backend
Data can be stored in prostresql probably to easily extract upvotes, etc…

CLEAN DATA

From our first assignment we learned how to clean data basically, so using NLKT https://www.nltk.org/  for lemmatization, spacy  https://spacy.io/ for tokenization and some sort of text normalization sklearn
(https://www.digitalocean.com/community/tutorials/normalize-data-in-python).
Basically first step is to make words like car, cars, car’s into one car, that is lemmatization, tokenization is separating these words, then a very time consuming part would be annotating the data in order to be able to train our model correctly. However we can use Distant Supervision where we just accept highly upvoted comments, or comments with a lot of replies etc…. 


TRAIN AI


Since we have other things to do and we don’t want to spend too much time training/annotating data we are going to generate weak labels using heuristics, there are three options for this, one we use the upvotes and stuff I mentioned earlier to allow a weather a sentence (discussion post) goes thru or not. The other is we us https://pypi.org/project/pytextrank/ were if  the sentences of posts are repeated throughout the discussion it will assume that it is of importance, and another way is using repeated words throughout the post, we then grab a sentence for example if all of these agree that the sentence/comment is valuable we keep it in for training We can also review a couple of them ourselves but honestly I don’t want to.


FRONT END (optional)

Make a Chrome plugin which grabs the website you're in, feeds it to our model in the back end and then summarizes it for the user, which then is displayed back in the front-end. 
