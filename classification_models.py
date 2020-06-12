"""

This class includes multiple models for text classification.

User can use any of the classifier from the following:

> SGD Classifier

"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

__author__ = "Talha Saqib"

# Third-party Imports
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2


class Classifier(object):

    # CLASSIFIERS
    @staticmethod
    def get_sgd_classifier():
        model = SGDClassifier(alpha=1e-3, random_state=42, max_iter=10, tol=None)
        return model

    @staticmethod
    def get_mnb_classifier():
        model = MultinomialNB()
        return model

    @staticmethod
    def get_logistic_classifier():
        model = LogisticRegression(n_jobs=1, C=1e5)
        return model

    @staticmethod
    def get_random_forest_classifier():
        model = RandomForestClassifier(n_estimators=30, criterion='gini', random_state=22,
                                       min_samples_split=5)
        return model

    # PIPELINES
    @staticmethod
    def get_pipeline(vectorizer, classifier, logger):
        try:
            pipeline = Pipeline([('vect', vectorizer),
                                 ('chi', SelectKBest(chi2, k=20000)),
                                 ('clf', classifier)
                                 ])
            logger.info("Pipeline Created")
            return pipeline
        except Exception as e:
            logger.error(e)

    @staticmethod
    def print_features_via_pipeline(pipeline, logger):
        try:
            # Accessing Vectorizer
            pipe = pipeline.named_steps['vect']
            features = pipe.get_feature_names()
            features_count = len(features)
            print("Number of textual features: ", features_count)
            # print("Textual Features: ", features)
            # pd.DataFrame(features).to_csv("Features.csv")

            # Accessing SelectKBest
            # pipe = pipeline.named_steps['chi']
            # mask = pipe.get_support()
            # selected_features = []
            # for bool, feature in zip(mask, features):
            #     if bool:
            #         selected_features.append(feature)
            # print("No. of Selected Features: ", len(selected_features))
            # print("Selected Features: ", selected_features)

        except Exception as e:
            logger.error(e)

    @staticmethod
    def get_pipeline_transfer_learn(vect1, vect2, classifier, logger):
        try:
            feature_union = ('feature_union', FeatureUnion([
                            ('vect1', vect1),
                            ('vect2', vect2),
                            ]))
            pipeline_both = Pipeline(steps=
                            [feature_union,
                            ('classifier', classifier)
                            ])

            logger.info("Pipleline Created")
            return pipeline_both
        except Exception as e:
            logger.error(e)