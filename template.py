"""
Foundations of Natural Language Processing

Assignment 1

Please complete functions, based on their doc_string description
and instructions of the assignment. 

To test your code run:

```
[hostname]s1234567 python3 s1234567.py
```

Before submission executed your code with ``--answers`` flag
```
[hostname]s1234567 python3 s1234567.py --answers
```
include generated answers.py file.

Best of Luck!
"""
from collections import defaultdict, Counter

import numpy as np  # for np.mean() and np.std()
import nltk, sys, inspect
import nltk.corpus.util
from nltk import MaxentClassifier
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import brown, ppattach, wordnet  # import corpora

# Import the Twitter corpus and LgramModel
from nltk_model import *  # See the README inside the nltk_model folder for more information

# Import the Twitter corpus and LgramModel
from twitter.twitter import *

twitter_file_ids = "20100128.txt"
assert twitter_file_ids in xtwc.fileids()


# Some helper functions

def ppEandT(eAndTs):
    '''
    Pretty print a list of entropy-tweet pairs

    :type eAndTs: list(tuple(float,list(str)))
    :param eAndTs: entropies and tweets
    :return: None
    '''

    for entropy, tweet in eAndTs:
        print("{:.3f} [{}]".format(entropy, ", ".join(tweet)))


def compute_accuracy(classifier, data):
    """
    Computes accuracy (range 0 - 1) of a classifier.
    :type classifier: NltkClassifierWrapper or NaiveBayes
    :param classifier: the classifier whose accuracy we compute.
    :type data: list(tuple(list(any), str))
    :param data: A list with tuples of the form (list with features, label)
    :rtype float
    :return accuracy (range 0 - 1).
    """
    correct = 0
    for d, gold in data:
        predicted = classifier.classify(d)
        correct += predicted == gold
    return correct / len(data)


def apply_extractor(extractor_f, data):
    """
    Helper function:
    Apply a feature extraction method to a labeled dataset.
    :type extractor_f: (str, str, str, str) -> list(any)
    :param extractor_f: the feature extractor, that takes as input V, N1, P, N2 (all strings) and returns a list of features
    :type data: list(tuple(str))
    :param data: a list with tuples of the form (id, V, N1, P, N2, label)

    :rtype list(tuple(list(any), str))
    :return a list with tuples of the form (list with features, label)
    """
    r = []
    for d in data:
        r.append((extractor_f(*d[1:-1]), d[-1]))
    return r


class NltkClassifierWrapper:
    """
    This is a little wrapper around the nltk classifiers so that we can interact with them
    in the same way as the Naive Bayes classifier.
    """

    def __init__(self, classifier_class, train_features, **kwargs):
        """

        :type classifier_class: a class object of nltk.classify.api.ClassifierI
        :param classifier_class: the kind of classifier we want to create an instance of.
        :type train_features: list(tuple(list(any), str))
        :param train_features: A list with tuples of the form (list with features, label)
        :param kwargs: additional keyword arguments for the classifier, e.g. number of training iterations.
        :return None
        """
        self.classifier_obj = classifier_class.train(
            [(NltkClassifierWrapper.list_to_freq_dict(d), c) for d, c in train_features], **kwargs)

    @staticmethod
    def list_to_freq_dict(d):
        """
        :param d: list(any)
        :param d: list of features
        :rtype dict(any, int)
        :return: dictionary with feature counts.
        """
        return Counter(d)

    def classify(self, d):
        """
        :param d: list(any)
        :param d: list of features
        :rtype str
        :return: most likely class
        """
        return self.classifier_obj.classify(NltkClassifierWrapper.list_to_freq_dict(d))

    def show_most_informative_features(self, n=10):
        self.classifier_obj.show_most_informative_features(n)


# End helper functions

# ==============================================
# Section I: Language Identification [60 marks]
# ==============================================

# Question 1 [7 marks]
def train_LM(corpus):
    '''
    Build a bigram letter language model using LgramModel
    based on the all-alpha subset the entire corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :rtype: LgramModel
    :return: A padded letter bigram model based on nltk.model.NgramModel
    '''

    # subset the corpus to only include all-alpha tokens,
    # converted to lower-case (_after_ the all-alpha check)
    corpus_tokens = []

    for word in corpus.words():
        if (word.isalpha()):
            corpus_tokens.append(word.lower())

    # Return a smoothed (using the default estimator) padded bigram
    # letter language model
    return LgramModel(2, corpus_tokens, pad_left=True, pad_right=True)


# Question 2 [7 marks]
def tweet_ent(file_name, bigram_model):
    '''
    Using a character bigram model, compute sentence entropies
    for a subset of the tweet corpus, removing all non-alpha tokens and
    tweets with less than 5 all-alpha tokens, then converted to lowercase

    :type file_name: str
    :param file_name: twitter file to process
    :rtype: list(tuple(float,list(str)))
    :return: ordered list of average entropies and tweets'''

    # Clean up the tweet corpus to remove all non-alpha
    # tokens and tweets with less than 5 (remaining) tokens, converted
    # to lowercase
    list_of_tweets = xtwc.sents(file_name)

    cleaned_list_of_tweets = []

    for tweet in list_of_tweets:
        alpha_tweet = []
        token_count = 0

        # remove non-alpha
        for word in tweet:
            if (word.isalpha()):
                alpha_tweet.append(word.lower())
                token_count += 1
        # remove <5 tokens
        if (token_count >= 5):
            cleaned_list_of_tweets.append(alpha_tweet)

    entro_tweet_pairs = []

    for tweet in cleaned_list_of_tweets:
        tweet_entro = 0
        for word in tweet:
            word_entro = bigram_model.entropy(word, True, True, False, True)
            tweet_entro += word_entro

        entro_tweet_pairs.append((tweet_entro / len(tweet), tweet))

    entro_tweet_pairs.sort()

    return entro_tweet_pairs

    # Construct a list of tuples of the form: (entropy,tweet)
    #  for each tweet in the cleaned corpus, where entropy is the
    #  average word for the tweet, and return the list of
    #  (entropy,tweet) tuples sorted by entropy


# Question 3 [8 marks]
def open_question_3():
    '''
    Question: What differentiates the beginning and end of the list
    of tweets and their entropies?

    :rtype: str
    :return: your answer [500 chars max]
    '''
    return inspect.cleandoc("""The main reason for the huge differences between entropies is the model we used, 
    which is trained in English corpus. The model is confident in predicting tweets in English but struggling 
    to predict non-English tweets. By inspecting the first 100 and last 100 tweets, it shows that tweets with low 
    entropies are made up of English words. In contrast, tweets with high entropies are made up of non-English words,
    typically Japanese in this corpus.""")[0:500]


# Question 4 [8 marks]
def open_question_4() -> str:
    '''
    Problem: noise in Twitter data

    :rtype: str
    :return: your answer [500 chars max]
    '''
    return inspect.cleandoc("""-- Using isalpha() will keep many non-English words. We could use isascii() to filter words instead of isalpha().
    -- There are many URLs, mentions and hashtags inside he data. Could replace them using re package. Replace (http\S*) | (@\S*) | (#\S*) with empty string.
    -- There are many misspellings in the data, we should correct them before doing further training.
    -- Some punctuation marks can be removed (e.g. "*" and "-" in "*Fad-ing*" by replacing [-;):(*] with empty string.""")[
           0:500]


# Question 5 [15 marks]
def tweet_filter(list_of_tweets_and_entropies):
    '''
    Compute entropy mean, standard deviation and using them,
    likely non-English tweets in the all-ascii subset of list 
    of tweets and their letter bigram entropies

    :type list_of_tweets_and_entropies: list(tuple(float,list(str)))
    :param list_of_tweets_and_entropies: tweets and their
                                    english (brown) average letter bigram entropy
    :rtype: tuple(float, float, list(tuple(float,list(str)), list(tuple(float,list(str)))
    :return: mean, standard deviation, ascii tweets and entropies,
             non-English tweets and entropies
    '''

    # Find the "ascii" tweets - those in the lowest-entropy 90%
    #  of list_of_tweets_and_entropies
    len_90_percent = int(len(list_of_tweets_and_entropies) * 0.9)
    list_of_ascii_tweets_and_entropies = list_of_tweets_and_entropies[0:len_90_percent]

    # Extract a list of just the entropy values
    list_of_entropies = []

    for (entropy, tweet) in list_of_ascii_tweets_and_entropies:
        list_of_entropies.append(entropy)

    # Compute the mean of entropy values for "ascii" tweets
    mean = np.mean(list_of_entropies)

    # Compute their standard deviation
    standard_deviation = np.std(list_of_entropies)

    # Get a list of "probably not English" tweets, that is
    #  "ascii" tweets with an entropy greater than (mean + std_dev))
    threshold = mean + standard_deviation
    list_of_not_English_tweets_and_entropies = []
    for (entropy, tweet) in list_of_ascii_tweets_and_entropies:
        if (entropy > threshold):
            list_of_not_English_tweets_and_entropies.append((entropy, tweet))

    # Return mean, standard_deviation,
    #  list_of_ascii_tweets_and_entropies,
    #  list_of_not_English_tweets_and_entropies
    return mean, standard_deviation, list_of_ascii_tweets_and_entropies, list_of_not_English_tweets_and_entropies


# Question 6 [15 marks]
def open_question_6():
    """
    Suppose you are asked to find out what the average per word entropy of English is.
    - Name 3 problems with this question, and make a simplifying assumption for each of them.
    - What kind of experiment would you perform to estimate the entropy after you have these simplifying assumptions?
       Justify the main design decisions you make in your experiment.
    :rtype: str
    :return: your answer [1000 chars max]
    """
    return inspect.cleandoc("""1 The average per word entropy of English is hard to be defined by estimating only few 
    corpora. Assumption: Compute per word entropy of many corpora and average them to approximate the average per word 
    entropy of English.
    2 Different models will give different result of the word entropy. It is hard to find the average per word entropy
     by using only one models. Assumption: Pick few models that could give the best result in computing then average them 
     to approximate the average per word entropy.
    3 We compute the per word cross-entropy. The cross-entropy must larger or equal to the true entropy. 
    Assumption: the cross-entropy equals to the true entropy. 
    - Pick 10 corpora from different domains and pick 5 language models.For each corpus, the per-word entropy of each 
    model is calculated, then average the best three models to obtain the per word entropy of the corpus. Next,the 
    per word entropy of each corpus is summed and then averaged to obtain the final per-word entropy of English.""")[
           :1000]


#############################################
# SECTION II - RESOLVING PP ATTACHMENT AMBIGUITY
#############################################

# Question 7 [15 marks]
class NaiveBayes:
    """
    Naive Bayes model with Lidstone smoothing (parameter alpha).
    """

    def __init__(self, data, alpha):
        """
        :type data: list(tuple(list(any), str))
        :param data: A list with tuples of the form (list with features, label)
        :type alpha: float
        :param alpha: \alpha value for Lidstone smoothing
        """
        self.vocab = self.get_vocab(data)
        self.alpha = alpha
        self.prior, self.likelihood = self.train(data, alpha, self.vocab)

    @staticmethod
    def get_vocab(data):
        """
        Compute the set of all possible features from the (training) data.
        :type data: list(tuple(list(any), str))
        :param data: A list with tuples of the form (list with features, label)
        :rtype: set(any)
        :return: The set of all features used in the training data for all classes.
        """
        vocab = set()

        for (features, label) in data:
            for feature in features:
                vocab.add(feature)

        return vocab

    @staticmethod
    def train(data, alpha, vocab):
        """
        Estimates the prior and likelihood from the data with Lidstone smoothing.

        :type data: list(tuple(list(any), str))
        :param data: A list of tuples ([f1, f2, ... ], c) with the first element
                     being a list of features and the second element being its class.

        :type alpha: float
        :param alpha: \alpha value for Lidstone smoothing

        :type vocab: set(any)
        :param vocab: The set of all features used in the training data for all classes.


        :rtype: tuple(dict(str, float), dict(str, dict(any, float)))
        :return: Two dictionaries: the prior and the likelihood (in that order).
        We expect the returned values to relate as follows to the probabilities:
            prior[c] = P(c)
            likelihood[c][f] = P(f|c)
        """
        assert alpha >= 0.0

        # Compute raw frequency distributions
        classes = []

        for (features, c) in data:
            classes.append(c)

        occurrences = Counter(classes)

        # Compute prior (MLE). Compute likelihood with smoothing.
        prior = dict()
        for c in occurrences:
            prior[c] = occurrences[c] / len(data)

        # Compute P(f|c)
        likelihood = dict()
        for c in prior:
            likelihood[c] = dict()
            for feature in vocab:
                count_feature = 0
                count_total_features = 0
                for (features, label) in data:
                    if (label == c):
                        counter = Counter(features)
                        count_feature += counter[feature]
                        count_total_features += len(features)
                likelihood[c][feature] = (count_feature + alpha) / (count_total_features + alpha * len(vocab))

        return (prior, likelihood)

    def prob_classify(self, d):
        """
        Compute the probability P(c|d) for all classes.
        :type d: list(any)
        :param d: A list of features.
        :rtype: dict(str, float)
        :return: The probability p(c|d) for all classes as a dictionary.
        """
        p_c_given_d = dict()

        p_d = 0

        for c in self.prior:
            p_d_given_c = 1
            for feature in d:
                if (feature in self.likelihood[c]):
                    p_d_given_c *= self.likelihood[c][feature]
            p_d += self.prior[c] * p_d_given_c

        for c in self.prior:
            p_d_given_c = 1
            for feature in d:
                if (feature in self.likelihood[c]):
                    p_d_given_c *= self.likelihood[c][feature]

            p_c_given_d[c] = p_d_given_c * self.prior[c] / p_d

        return p_c_given_d


    def classify(self, d):
        """
        Compute the most likely class of the given "document" with ties broken arbitrarily.
        :type d: list(any)
        :param d: A list of features.
        :rtype: str
        :return: The most likely class.
        """
        prob = self.prob_classify(d)
        max = 0
        max_label = None

        for x in prob:
            if (prob[x] > max):
                max = prob[x]
                max_label = x

        return max_label


# Question 8 [10 marks]
def open_question_8() -> str:
    """
    How do you interpret the differences in accuracy between the different ways to extract features?
    :rtype: str
    :return: Your answer of 500 characters maximum.
    """
    return inspect.cleandoc("""-Features might common in one class and rare in another. The model 
    can easily classify class by monitoring these features. However, some features may be common in both classes. 
    In this case, the model is not confident in predicting the input labels by inspecting these features.
    -The accuracy of Q7 is 79%, less than logistic regression. Some features provide more information 
    than others. Logistic regression could assign different weights for features.
    While naive bayes has same weight.""")[:500]


# Feature extractors used in the table:
# see your_feature_extractor for documentation on arguments and types.
def feature_extractor_1(v, n1, p, n2):
    return [v]


def feature_extractor_2(v, n1, p, n2):
    return [n1]


def feature_extractor_3(v, n1, p, n2):
    return [p]


def feature_extractor_4(v, n1, p, n2):
    return [n2]


def feature_extractor_5(v, n1, p, n2):
    return [("v", v), ("n1", n1), ("p", p), ("n2", n2)]


# Question 9.1 [5 marks]
def your_feature_extractor(v, n1, p, n2):
    """
    Takes the head words and produces a list of features. The features may
    be of any type as long as they are hashable.
    :type v: str
    :param v: The verb.
    :type n1: str
    :param n1: Head of the object NP.
    :type p: str
    :param p: The preposition.
    :type n2: str
    :param n2: Head of the NP embedded in the PP.
    :rtype: list(any)
    :return: A list of features produced by you.
    """
    lemmatizer = WordNetLemmatizer()

    # v = lemmatizer.lemmatize(v)
    n1 = lemmatizer.lemmatize(n1)
    # n2 = lemmatizer.lemmatize(n2)

    return [("v", v), ("p", p), ("v+p", v + p), ("n1+p", n1 + p), ("n2+p", n2 + p)]


# Question 9.2 [10 marks]
def open_question_9():
    """
    Briefly describe your feature templates and your reasoning for them.
    Pick 3 examples of informative features and discuss why they make sense or why they do not make sense
    and why you think the model relies on them.
    :rtype: str
    :return: Your answer of 1000 characters maximum.
    """
    return inspect.cleandoc("""...""")[:1000]


"""
Format the output of your submission for both development and automarking. 
!!!!! DO NOT MODIFY THIS PART !!!!!
"""


def answers():
    # Global variables for answers that will be used by automarker
    global ents, lm
    global best10_ents, worst10_ents, mean, std, best10_ascci_ents, worst10_ascci_ents
    global best10_non_eng_ents, worst10_non_eng_ents
    global answer_open_question_4, answer_open_question_3, answer_open_question_6, \
        answer_open_question_8, answer_open_question_9
    global ascci_ents, non_eng_ents

    global naive_bayes
    global acc_extractor_1, naive_bayes_acc, lr_acc, logistic_regression_model, dev_features

    print("*** Part I***\n")

    print("*** Question 1 ***")
    print('Building brown bigram letter model ... ')
    lm = train_LM(brown)
    print('Letter model built')

    print("*** Question 2 ***")
    ents = tweet_ent(twitter_file_ids, lm)
    print("Best 10 english entropies:")
    best10_ents = ents[:10]
    ppEandT(best10_ents)
    print("Worst 10 english entropies:")
    worst10_ents = ents[-10:]
    ppEandT(worst10_ents)

    print("*** Question 3 ***")
    answer_open_question_3 = open_question_3()
    print(answer_open_question_3)

    print("*** Question 4 ***")
    answer_open_question_4 = open_question_4()
    print(answer_open_question_4)

    print("*** Question 5 ***")
    mean, std, ascci_ents, non_eng_ents = tweet_filter(ents)
    print('Mean: {}'.format(mean))
    print('Standard Deviation: {}'.format(std))
    print('ASCII tweets ')
    print("Best 10 English entropies:")
    best10_ascci_ents = ascci_ents[:10]
    ppEandT(best10_ascci_ents)
    print("Worst 10 English entropies:")
    worst10_ascci_ents = ascci_ents[-10:]
    ppEandT(worst10_ascci_ents)
    print('--------')
    print('Tweets considered non-English')
    print("Best 10 English entropies:")
    best10_non_eng_ents = non_eng_ents[:10]
    ppEandT(best10_non_eng_ents)
    print("Worst 10 English entropies:")
    worst10_non_eng_ents = non_eng_ents[-10:]
    ppEandT(worst10_non_eng_ents)

    print("*** Question 6 ***")
    answer_open_question_6 = open_question_6()
    print(answer_open_question_6)

    print("*** Part II***\n")

    print("*** Question 7 ***")
    naive_bayes = NaiveBayes(apply_extractor(feature_extractor_5, ppattach.tuples("training")), 0.1)
    naive_bayes_acc = compute_accuracy(naive_bayes, apply_extractor(feature_extractor_5, ppattach.tuples("devset")))
    print(f"Accuracy on the devset: {naive_bayes_acc * 100}%")

    print("*** Question 8 ***")
    answer_open_question_8 = open_question_8()
    print(answer_open_question_8)

    # This is the code that generated the results in the table of the CW:

    # A single iteration of suffices for logistic regression for the simple feature extractors.
    #
    # extractors_and_iterations = [feature_extractor_1, feature_extractor_2, feature_extractor_3, feature_extractor_4, feature_extractor_5]
    #
    # print("Extractor    |  Accuracy")
    # print("------------------------")
    #
    # for i, ex_f in enumerate(extractors_and_iterations, start=1):
    #     training_features = apply_extractor(ex_f, ppattach.tuples("training"))
    #     dev_features = apply_extractor(ex_f, ppattach.tuples("devset"))
    #
    #     a_logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=6, trace=0)
    #     lr_acc = compute_accuracy(a_logistic_regression_model, dev_features)
    #     print(f"Extractor {i}  |  {lr_acc*100}")
    #
    #
    # print("*** Question 9 ***")
    # training_features = apply_extractor(your_feature_extractor, ppattach.tuples("training"))
    # dev_features = apply_extractor(your_feature_extractor, ppattach.tuples("devset"))
    # logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=10)
    # lr_acc = compute_accuracy(logistic_regression_model, dev_features)
    #
    # print("30 features with highest absolute weights")
    # logistic_regression_model.show_most_informative_features(30)
    #
    # print(f"Accuracy on the devset: {lr_acc*100}")
    #
    # answer_open_question_9 = open_question_9()
    # print("Answer to open question:")
    # print(answer_open_question_9)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--answers':
        from autodrive_embed import run, carefulBind
        import adrive1

        with open("userErrs.txt", "w") as errlog:
            run(globals(), answers, adrive1.extract_answers, errlog)
    else:
        answers()
