"""

This class is for text pre-processing.

It includes:
> Non-alphabets filter
> Stemmer
> Lemmatizer

"""
import pandas as pd

__author__ = "Talha Saqib"

# Third-party Imports
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import WhitespaceTokenizer
import nltk.corpus as corpus
from nltk.corpus import stopwords
from textblob import TextBlob


class TextNormalizer(object):

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = WhitespaceTokenizer()
        self.stemmer = PorterStemmer()
        self.vocabulary = set(corpus.words.words())
        self.stop_words = list(stopwords.words('english'))

    # ON DATAFRAME
    def filter_special_chars(self, column, logger):
        try:
            column = column.str.replace('[^\w\s]', '')
            logger.info("Special characters removed")
            return column
        except Exception as e:
            logger.error(e)

    def filter_numbers(self, column, logger):
        try:
            column = column.str.replace('[0-9]', '')
            logger.info("Numbers removed")
            return column
        except Exception as e:
            logger.error(e)

    def lower_case(self, text):
        try:
            return " ".join([word.lower() for word in text.split()])
        except Exception as e:
            print("Error: ", e)

    def remove_stop_words(self, text):
        try:
            return " ".join([word for word in text.split() if word not in self.stop_words])
        except Exception as e:
            print("Error: ", e)

    def remove_non_english_words(self, text):
        try:
            return " ".join([word for word in text.split() if word.lower() in self.vocabulary])
        except Exception as e:
            print("Error: ", e)

    def lemmatize_text(self, text):
        try:
            return " ".join([self.lemmatizer.lemmatize(word.decode('utf-8')) for word in text.split()])
        except Exception as e:
            print("Error: ", e)

    def stem_text(self, text):
        try:
            return " ".join([self.stemmer.stem(word.decode('utf-8')) for word in text.split()])
        except Exception as e:
            print("Error: ", e)

    def correct_spelling(self, text):
        try:
            return str(TextBlob(text).correct())
        except Exception as e:
            print("Error: ", e)

    def text_preprocessing(self, column, logger):
        try:
            # Filtering special characters
            column = self.filter_special_chars(column, logger)

            # Filtering numbers
            # column = self.filter_numbers(column, logger)

            # Lowercasing words
            column = column.apply(self.lower_case)
            logger.info("Words Lowercased")

            # Filtering stop words
            column = column.apply(self.remove_stop_words)
            logger.info("Stop words removed")

            # Filtering non-english words
            # column = column.apply(self.remove_non_english_words)
            # logger.info("Non-English words removed")

            # Applying Lemmatization
            # column = column.apply(self.lemmatize_text)
            # logger.info("Lemmatization Done")

            # Applying Stemming
            column = column.apply(self.stem_text)
            logger.info("Stemming Done")

            return column
        except Exception as e:
            logger.error(e)

    # ON LIST OF WORDS
    def stem_list_of_words(self, list_of_words, logger):
        try:
            stemmed_words_list = list(set([self.stemmer.stem(word.decode('utf-8')) for word in list_of_words]))
            logger.info("Words List Stemmed")
            return stemmed_words_list
        except Exception as e:
            logger.error(e)

    def lemm_list_of_words(self, list_of_words, logger):
        try:
            stemmed_words_list = list(set([self.lemmatizer.lemmatize(word.decode('utf-8')) for word in list_of_words]))
            logger.info("Words List Lemmatized")
            return stemmed_words_list
        except Exception as e:
            logger.error(e)

    def lowercase_list_of_words(self, list_of_words, logger):
        try:
            lowercased_words = list(set([word.lower() for word in list_of_words]))
            logger.info("Words List Lowercased")
            return lowercased_words
        except Exception as e:
            logger.error(e)

    def text_preprocessing_list(self, list_of_words, logger):
        try:
            list_of_words = self.lowercase_list_of_words(list_of_words, logger)
            list_of_words = self.stem_list_of_words(list_of_words, logger)
            # list_of_words = self.lemm_list_of_words(list_of_words, logger)

            logger.info("Text Preprocessing Done on List")
            return list_of_words
        except Exception as e:
            logger.error(e)

    # TEXT HANDLERS
    @staticmethod
    def get_most_occurring_words(strings_column, no_of_words, logger):
        try:
            freq = pd.Series(' '.join(strings_column).split()).value_counts()[:no_of_words]
            freq = pd.DataFrame(freq).reset_index()
            freq.columns = ["Words", "Frequency"]
            print(freq)
            logger.info("Frequent Words Retrieved")
            return freq
        except Exception as e:
            logger.error(e)

    # VECTORIZERS
    @staticmethod
    def get_count_vectorizer():
        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 4))
        return vectorizer

    @staticmethod
    def get_tfidf_vectorizer():
        vectorizer = TfidfVectorizer(use_idf=True, stop_words='english', ngram_range=(1, 4))
        return vectorizer

    @staticmethod
    def vectorize_text(vectorizer, text_column, logger):
        try:
            x_train = vectorizer.fit_transform(text_column)

            print("Number of textual features: ", len(vectorizer.get_feature_names()))
            # print("Textual Features: ", vectorizer.get_feature_names())

            logger.info("Vectorization Done")
            return x_train

        except Exception as e:
            logger.error(e)