import nltk.data
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def calculate_years_worked():
    '''shows user how many full years of their
    lives they've spent working'''

    start_date = input("What year (YYYY) did you start working full-time? ")
    hours_worked = input("\nApproximately how many hours do you work in a week? ")
    total_years = ((int(hours_worked)) * 4 * 12 * (2017 - int(start_date)))/8760
    total_years = round(total_years, 2) # rounds up to two decimal places
    print("\nRoughly speaking, you've spent a full {} years of your life behind a desk, \nor wherever it is you work.".format(total_years))

def get_job_satisfaction():
    '''ranks a user's feelings towards work on
    a 1-10 scale and recommends online reading
    should they feel too much love for it'''

    # perform sentiment analysis on user input
    print("\nWrite a short paragraph describing how you feel about work. Be descriptive: \nwhat do you enjoy or dislike?")

    sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    text = input("\nAnswer: ")
    sents = sent_tokenizer.tokenize(text)
    result = { 'compound':[] }

    for sent in sents:
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(sent)

        result['compound'].append(vs['compound'])

    # get compound score from sentiment analysis to determine job satisfaction
    mean_score = sum(result['compound'])/len(result['compound'])
    mean_score = round(mean_score, 2) # return the floating point value number rounded to 2 digits after the decimal point
    print("\nYour compound score is {}: -1 is the most extreme negative \nand +1 is the most extreme positive.".format(mean_score))

    # graph sentence by sentence, with mean score in red
    plt.bar(list(range(len(sents))), result['compound'])
    plt.title('Compound Score by Sentence')
    plt.axhline(y = mean_score, xmin = 0, xmax = len(sents), c='r')
    plt.show()

    # find satisfaction_decile, i.e. 1-10 value
    score_range_vals = [-1.0, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, 1.0]
    for idx, val in enumerate(score_range_vals): # takes an iterable (like a list) and returns index & value of ea. element
        if val <= mean_score < score_range_vals[idx + 1]:
            satisfaction_decile = idx + 1

    score_description_dict = {1: "HATE", 2: "HATE", 3: "really don't like", 4: "don't like", 5: "feel whatevs about", 6: "feel whatevs about", 7: "like", 8: "really like", 9: "LOVE", 10: "LOVE"}

    # summarize score and feelings
    print("\nOn a scale of 1-10, you enjoy your job a {}.".format(satisfaction_decile))
    print("\nIn other words, you {} your job...".format(score_description_dict[satisfaction_decile]))
    if satisfaction_decile > 6:
        print("\nHowever, this can be problematic. Here's some suggested reading: \nhttp://www.jacobinmag.com/2014/01/in-the-name-of-love")


if __name__ == '__main__':
    calculate_years_worked()
    get_job_satisfaction()
