"""
This class is for classifying a company as:

> B2B
> B2C
> Both
> None

via machine learning models, provided the labeled data set of different companies.

"""

__author__ = "Talha Saqib"

# Local Imports
from data_manipulation import DataManipulator
from classification_models import Classifier
from text_normalizer import TextNormalizer
import pickle
from collections import Counter

# Third-party Imports
from sklearn.model_selection import train_test_split
from configurator import Configurator
import pandas as pd


class BusinessTypeClassifier(object):
    config_parser = None
    logger = None

    def __init__(self):
        # Setting general utilities
        configurator = Configurator()
        self.config_parser, self.logger = configurator.set_configurator()

        self.vectorizer = TextNormalizer.get_tfidf_vectorizer()
        self.classifier = Classifier.get_random_forest_classifier()

    def oversample_after_split(self, fields, x_train, y_train):
        try:
            train_data = pd.DataFrame({fields[0]: x_train, fields[1]: y_train})
            train_data_re = DataManipulator.oversample_data(train_data, fields[1], self.logger)
            x_train = train_data_re[fields[0]]
            y_train = train_data_re[fields[1]]
            return x_train, y_train
        except Exception as e:
            self.logger.error(e)

    def extract_freq_words_for_each_label(self, x_train, y_train, fields, logger):
        try:
            df = pd.DataFrame({fields[0]: x_train, fields[1]: y_train})

            d1 = df[df[fields[1]] == 1]
            freq = TextNormalizer.get_most_occurring_words(d1[fields[0]], 1000, logger)
            freq.to_csv("most_occuring_words_for_b2b.csv")

            d2 = df[df[fields[1]] == 2]
            freq = TextNormalizer.get_most_occurring_words(d2[fields[0]], 1000, logger)
            freq.to_csv("most_occuring_words_for_b2c.csv")

            d3 = df[df[fields[1]] == 3]
            freq = TextNormalizer.get_most_occurring_words(d3[fields[0]], 1000, logger)
            freq.to_csv("most_occuring_words_for_both.csv")

        except Exception as e:
            logger.error(e)

    def write_output(self, x_test, y_test, y_pred):
        try:
            data = pd.DataFrame()
            data['Company Decscription'] = x_test
            data['Actual Labels'] = y_test
            data['Predicted Labels'] = y_pred

            data.to_csv("output.csv")
            self.logger.info("File Written")
        except Exception as e:
            self.logger.error(e)

    def train_model(self, x_train, y_train):
        try:
            # Without pipeline
            # x_train = TextNormalizer.vectorize_text(self.vectorizer, x_train, logger)
            # self.classifier.fit(x_train, y_train)
            # model = self.classifier

            # With pipeline
            self.pipeline = Classifier.get_pipeline(self.vectorizer, self.classifier, self.logger)
            self.pipeline.fit(x_train, y_train)
            model = self.pipeline

            # save the model to disk
            # filename = 'b2b_b2c_merged_100_RF_ov_model.sav'
            # pickle.dump(model, open(filename, 'wb'))

            self.logger.info("Model Trained")
        except Exception as e:
            self.logger.error(e)

    def classify(self, x_test, y_test):
        try:
            # Without pipeline
            # x_test = self.vectorizer.transform(x_test)
            # y_pred = self.classifier.predict(x_test)

            # With pipeline
            y_pred = self.pipeline.predict(x_test)

            DataManipulator.evaluate(y_test, y_pred, self.logger)
            # self.write_output(x_test, y_test, y_pred)
        except Exception as e:
            self.logger.error(e)

    def classify_with_saved_model(self, x_test, y_test, filename):
        try:
            # load the model from disk
            model = pickle.load(open(filename, 'rb'))
            self.logger.info("Model loaded")

            y_pred = model.predict(x_test)

            DataManipulator.evaluate(y_test, y_pred, self.logger)
            # self.write_output(x_test, y_test, y_pred)
        except Exception as e:
            self.logger.error(e)

    # For two features
    def train_model_2(self, x_train, y_train, fields):
        try:
            self.c1 = self.vectorizer.fit(x_train[fields[0]], self.logger)
            self.c2 = self.vectorizer.fit(x_train[fields[2]], self.logger)

            x_train_1 = pd.DataFrame(self.c1.transform(x_train[fields[0]]).todense(),
                                     columns=self.c1.get_feature_names())
            x_train_2 = pd.DataFrame(self.c2.transform(x_train[fields[2]]).todense(),
                                     columns=self.c2.get_feature_names())
            x_train = pd.concat([x_train_1, x_train_2], axis=1)

            # x_train = self.vectorizer.transform(x_train)
            self.classifier.fit(x_train, y_train)

            self.logger.info("Model Trained")
        except Exception as e:
            self.logger.error(e)

    def classify_2(self, x_test, y_test, fields):
        try:
            x_test_1 = pd.DataFrame(self.c1.transform(x_test[fields[0]]).todense(), columns=self.c1.get_feature_names())
            x_test_2 = pd.DataFrame(self.c2.transform(x_test[fields[2]]).todense(), columns=self.c2.get_feature_names())

            x_test = pd.concat([x_test_1, x_test_2], axis=1)

            y_pred = self.classifier.predict(x_test)

            DataManipulator.evaluate(y_test, y_pred, self.logger)
            # self.write_output(x_test, y_test, y_pred, logger)
        except Exception as e:
            self.logger.error(e)


def main():
    business_classifier = BusinessTypeClassifier()
    text_normalizer = TextNormalizer()

    logger = business_classifier.logger
    config_parser = business_classifier.config_parser

    FILE_SECTION = "filenames"
    filename_key = "merged"
    fields = ["Company_Description", "Class_Label"]
    filename_key_1 = "new"
    fields_1 = ["Company_Description", "Class_Label"]
    # filename_key = "unlabeled"
    # fields = ["CO:coDesc", "Fixed Final"]

    try:
        # Reading input data
        data = DataManipulator.read_data(FILE_SECTION, filename_key, logger, config_parser)
        data_1 = DataManipulator.read_data(FILE_SECTION, filename_key_1, logger, config_parser)

        # Filtering null rows
        data = DataManipulator.remove_nulldata(data[fields], logger)
        data_1 = DataManipulator.remove_nulldata(data_1[fields_1], logger)

        # Balance
        # data = data[data[fields[1]] != 0]
        # data_1 = data_1[data_1[fields_1[1]] != 0]

        # Normalizing Text
        data[fields[0]] = text_normalizer.text_preprocessing(data[fields[0]], logger)
        data_1[fields_1[0]] = text_normalizer.text_preprocessing(data_1[fields_1[0]], logger)

        # Splitting data into 70% training and 30% test data
        # x_train, x_test, y_train, y_test = train_test_split(
        #     data[fields[0]], data[fields[1]], test_size=0.3, random_state=42)

        # Without splitting
        x_train = data[fields[0]]
        y_train = data[fields[1]]
        x_test = data_1[fields_1[0]]
        y_test = data_1[fields_1[1]]

        # Oversampling after split
        x_train, y_train = business_classifier.oversample_after_split(fields, x_train, y_train)

        # business_classifier.extract_freq_words_for_each_label(x_train, y_train, fields, logger)

        business_classifier.train_model(x_train, y_train)
        business_classifier.classify(x_test, y_test)

        # filename = 'b2b_b2c_merged_100_RF_model.sav'
        # business_classifier.classify_with_saved_model(x_test, y_test, filename)

        # Multifeature
        # business_classifier.train_model_2(x_train, y_train, fields, logger)
        # business_classifier.classify_2(x_test, y_test, fields, logger)

    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    main()
