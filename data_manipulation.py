"""
This class contains functions to manipulate data as per requirements. It also includes data representation functions.

The methods include:

> Dataframes merging, row-wise
> Null rows deletion
> Reading CSV to dataframe
> Ploting bar chart of a column values
> Printing frequency of unique values in a numpy array
> Resample dataframe using SMOTE

"""

__author__ = "Talha Saqib"

# Local Imports
import csv

# Third-party imports
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report, confusion_matrix
from sklearn.feature_selection import chi2
from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer


# from imblearn.over_sampling import SMOTE


class DataManipulator(object):

    @staticmethod
    def evaluate(y_true, y_predicted, logger):
        try:
            print("ACCURACY: {}".format(accuracy_score(y_true, y_predicted) * 100))
            # print("PRECISION: {}".format(precision_score(y_true, y_predicted, average='weighted') * 100))
            # print("RECALL: {}".format(recall_score(y_true, y_predicted, average='weighted') * 100))
            print("F1-SCORE: {}".format(f1_score(y_true, y_predicted, average='weighted') * 100))

            error_matrix = pd.DataFrame(confusion_matrix(y_true, y_predicted))
            print(error_matrix)

            # print(classification_report(y_true, y_predicted))
            # Works only for binary labels
            # print("AUC-SCORE: {}".format(roc_auc_score(y_true, y_predicted, average='macro') * 100))
        except Exception as e:
            logger.log(e)

    @staticmethod
    def print_unique_frequency(nparray, logger):
        try:
            unique, counts = np.unique(nparray, return_counts=True)
            frequency = dict(zip(unique, counts))
            print(frequency)
        except Exception as e:
            logger.log(e)

    # DATAFRAME HANDLING
    @staticmethod
    def read_data(files_header, filename_key, logger, config_parser):
        try:
            filename = config_parser.get(files_header, filename_key)
            with open(filename) as file_handler:
                data = pd.read_csv(file_handler)
            logger.info("Reading Done")
            return data
        except Exception as e:
            logger.error(e)

    @staticmethod
    def dict_write_csv(filename, fields, dictionary, logger):
        try:
            with open(filename, 'w') as file_handler:
                writer = csv.DictWriter(file_handler, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerow(dictionary)
                logger.info("File Written")
        except Exception as e:
            logger.error(e)

    @staticmethod
    def merge_data(dataframe1, dataframe2, logger):
        try:
            merged_data = pd.concat([dataframe1, dataframe2], ignore_index=True, sort=False)
            logger.info("Data Merged")
            return merged_data
        except Exception as e:
            logger.error(e)

    @staticmethod
    def remove_nulldata(dataframe, logger):
        try:
            filtered_data = dataframe.dropna()
            logger.info("Null Data filtered")
            return filtered_data.reset_index(drop=True)
        except Exception as e:
            logger.error(e)

    # PLOT FUNCTIONS
    @staticmethod
    def plot_bar_chart(column, logger):
        try:
            column.value_counts().plot.bar()
            plt.show()
            logger.info("Plotting Done")
        except Exception as e:
            logger.error(e)

    @staticmethod
    def plot_most_useful_words(tvec, x_train, x_test, y_train, logger):
        try:
            x_train_tfidf = tvec.fit_transform(x_train)
            x_validation_tfidf = tvec.transform(x_test)
            chi2score = chi2(x_train_tfidf, y_train)[0]

            plt.figure(figsize=(15, 10))
            wscores = zip(tvec.get_feature_names(), chi2score)
            wchi2 = sorted(wscores, key=lambda x: x[1])
            topchi2 = zip(*wchi2[-100:])
            x = range(len(topchi2[1]))

            print("Most Useful Words: ", topchi2)
            pd.DataFrame(topchi2).to_csv("most_useful_words.csv")

            # labels = topchi2[0]
            # plt.barh(x, topchi2[1], align='center', alpha=0.2)
            # plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)
            # plt.yticks(x, labels)
            # plt.xlabel('$\chi^2$')
            # plt.show()
        except Exception as e:
            logger.error(e)

    # DATA NORMALIZATION
    @staticmethod
    def resample_data(data, field, no_of_rows, logger):
        try:
            print(data[field].value_counts())

            unique_labels = list(data[field].unique())

            dfs_to_sample = [data[data[field] == x] for x in unique_labels]

            oversampled_list = [resample(x, replace=True, n_samples=no_of_rows, random_state=27) for x in
                                dfs_to_sample]

            # Merging unsampled and sampled data, also suffeling rows
            oversampled_data = pd.concat(oversampled_list)
            oversampled_data = oversampled_data.sample(frac=1, random_state=14)

            print(oversampled_data[field].value_counts())

            logger.info("Over-Sampling Done")
            return oversampled_data
        except Exception as e:
            logger.error(e)

    @staticmethod
    def oversample_data(data, field, logger):
        try:
            print(data[field].value_counts())

            max_labels = list(data[field].mode())
            unique_labels = list(data[field].unique())
            labels_to_sample = [x for x in unique_labels if x not in max_labels]

            max_label_dfs = [data[data[field] == x] for x in max_labels]
            dfs_to_sample = [data[data[field] == x] for x in labels_to_sample]

            max_label_count = data[field].value_counts().max()
            sampled_list = [resample(x, replace=True, n_samples=max_label_count, random_state=27) for x in
                            dfs_to_sample]

            # Merging unsampled and sampled data, also suffeling rows
            oversampled_list = max_label_dfs + sampled_list
            oversampled_data = pd.concat(oversampled_list)
            oversampled_data = oversampled_data.sample(frac=1, random_state=14)

            print(oversampled_data[field].value_counts())

            logger.info("Over-Sampling Done")
            return oversampled_data
        except Exception as e:
            logger.error(e)

    @staticmethod
    def scale_features(features, logger):
        try:
            sc = StandardScaler()
            features = sc.fit_transform(features)
            logger.info("Features Scaled")
            return features
        except Exception as e:
            logger.error(e)

    @staticmethod
    def normalize_features(features, logger):
        try:
            nrm = Normalizer()
            features = nrm.fit_transform(features)
            logger.info("Features Normalized")
            return features
        except Exception as e:
            logger.error(e)

    # @staticmethod
    # def oversample_data_smote(features, labels, logger):
    #     # Only Python 3+ compatible
    #     # Working only with pipelined vector and tfidf and not any good!
    #     try:
    #         print(labels.value_counts())
    #
    #         sm = SMOTE(random_state=42)
    #         x_new, y_new = sm.fit_resample(features, labels)
    #         logger.info("SMOTE Resampling Done")
    #
    #         DataManipulator.print_unique_frequency(y_new, logger)
    #
    #         return x_new, y_new
    #     except Exception as e:
    #         logger.error(e)
