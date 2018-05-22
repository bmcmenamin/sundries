import os
import time
import datetime

import tweepy


def get_authorized_api():
    consumer_key = os.getenv('GB_CONSUMER_KEY')
    consumer_secret = os.getenv('GB_CONSUMER_SECRET')
    access_token = os.getenv('GB_ACCESS_TOKEN')
    access_token_secret = os.getenv('GB_TOKEN_SECRET')

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)
    return api


def rate_limiter(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            time.sleep(15 * 60)


def get_user_recent_tweets(api, id):
    for tweet in rate_limiter(tweepy.Cursor(api.user_timeline, id=id).items()):
        yield tweet


def search_messages(api, search_params):
    for tweet in rate_limiter(tweepy.Cursor(api.search, **search_params).items()):
        yield tweet


def search_and_reply(search_params, payload):

    api = get_authorized_api()
    me_account = api.me()

    previous_targets = set([
        tweet.in_reply_to_user_id
        for tweet in get_user_recent_tweets(api, me_account.id)
        if tweet.in_reply_to_user_id
    ])

    for tweet in search_messages(api, search_params):
        if tweet.author.id not in previous_targets:
            print("Tweeting at: {}".format(tweet.author.screen_name))
            api.update_status(
                payload.format(target=tweet.author.screen_name),
                in_reply_to_status_id=tweet.id
            )

            previous_targets.add(tweet.author.id)


if __name__ == '__main__':

    OLDEST_REPLY_DAYS = 14

    SEARCH_PARAMS = {
        'q': '"gummi\ bears\ theme\ song" -williams -toto',
        'lang': 'en',
        'since': (
            datetime.datetime.now() - datetime.timedelta(days=OLDEST_REPLY_DAYS)
        ).strftime('%Y-%m-%d')
    }

    PAYLOAD = (
        "@{target} i thought you might like to know that the "
        "Gummi Bears Theme Song was sung by John Williams' son."
    )

    search_and_reply(SEARCH_PARAMS, PAYLOAD)
