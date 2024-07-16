from flask import Flask, request, jsonify
from google.cloud import dialogflow
from datetime import datetime
from similiarity import recommandbySim
from filtering import filter
import re
import preprocessing

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>hello world</h1>"
l = 'Hello'

author = ""
genre = ""
rating = 0
@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()
    print(req['queryResult']['intent']['displayName'])
    type = req['queryResult']['intent']['displayName']
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time)
    if type == 'Default Welcome Intent':
        reText = req['queryResult']['fulfillmentText']
        print(reText)
        return {
            "fulfillmentText": f'{reText}',
            "source": "webhookdata"
        }
    if type == 'AuthorBasedRecommendation':
        reText = req['queryResult']['fulfillmentText']
        global author
        author = str(req['queryResult']['parameters']['author'])
        print(author)
        print(reText)
        return {
            "fulfillmentText": f'Your prefered author is {author}. {reText}',
            "source": "webhookdata"
        }
    if type == 'GenreBasedRecommendation':
        reText = req['queryResult']['fulfillmentText']
        global genre
        genre =  str(req['queryResult']['parameters']['genres'])
        print(genre)
        return {
            "fulfillmentText": f'Your prefered genre is {genre}. {reText}',
            "source": "webhookdata"
        }
    if type == 'FeedbackBasedRecommdation':
        reText = req['queryResult']['fulfillmentText']
        ratinglst =  req['queryResult']['parameters']['rating']
        global rating
        rating = int(ratinglst[0].split()[0])
        filter(genre, author, rating)
        return {
            "fulfillmentText": f'Your prefered rating is {rating} stars. {reText}',
            "source": "webhookdata"
        }
    if type == 'SimilarityBasedRecommendation':
        reText = req['queryResult']['fulfillmentText']
        genre =  req['queryResult']['parameters']['keywords']
        return {
            "fulfillmentText": f'{reText}: {recommandbySim(reText)}',
            "source": "webhookdata"
        }

if __name__ == '__main__':
    app.run(port=5125)
