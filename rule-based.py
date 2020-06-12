"""
This file implements rule-based b2b/b2c classification with the given rules (keywords for b2b and b2c descriptions)
"""

__author__ = "Talha Saqib"

# Local Imports
from configurator import Configurator
from data_manipulation import DataManipulator
from text_normalizer import TextNormalizer


class RuleBasedB2BB2C:
    config_parser = None
    logger = None

    def __init__(self):
        try:
            # Setting general utilities
            configurator = Configurator()
            self.config_parser, self.logger = configurator.set_configurator()

            self.text_normalizer = TextNormalizer()

            self.b2c_words = ['B2C', 'millions', 'million', 'entertainment', 'consumer', 'students', 'specialties', 'Health',
                         'Wellness', 'Google Play', 'App', 'apps', 'itune', 'parent', 'artists', 'retailer', 'supermarkets',
                         'men', 'women', 'Museums', 'Institutions', 'marketplace',
                         'residents', 'community', 'peer', 'IOS', 'Android', 'mobile application', 'exchange', 'families',
                         'Kids', 'Home', 'homes', 'store', 'consumers', 'home', 'cleaners', 'meal delivery',
                         'child', 'individuals', 'student', 'children', 'app', 'parent', 'sitter', 'investor',
                         'homeowners', 'consumers', 'friends', 'kids', 'journey', 'individuals', 'People']
            self.b2b_words = ['enterprises', 'CRM', 'businesses', 'B2B', 'enterprise', 'business', 'organisations', 'agency',
                         'agencies', 'businesses', 'companies', 'ENTERPRISE', 'credit unions', 'financial institutions',
                         'brands', 'healthcare systems', 'pharma companies', 'service centres', 'cloud providers',
                         'cloud communications', 'organisations', 'enterprise', 'businesses', 'clients', 'physicians',
                         'trucking company', 'brands', 'account holders', 'WordPress', 'companies',
                         'law firms', 'analytics platform', 'developers', 'APIs', 'employee', 'doctors', 'employees',
                         'organizations', 'sales', 'marketing', 'suppliers', 'biotechnology', 'teams', 'utilities',
                         'builders', 'sales', 'organizations', 'pubishers', 'broadcasters', 'classroom', 'teams', 'oracle',
                         'brand', 'service management', 'corporate', 'user experience', 'healthcare providers', 'engineers',
                         'designers', 'brand', 'consultancy', 'sales', 'teams', 'brand', 'operations software',
                         'employers', 'sellers', 'teams', 'cities', 'transportation', 'Teams', 'Agencies', 'Enterprise',
                         'API', 'restaurants', 'provider', 'teams', 'developers', 'institutions', 'managers', 'applicants',
                         'organizations', 'corporates', 'Project Management', 'DevOps', 'open-source', 'staff', 'employers',
                         'Infrastructure', 'charities', 'cyber', 'security',
                         'information services', 'service providers', 'dealerships', 'digital marketing',
                         'sales', 'marketers', 'marketing', 'platform', 'government', 'industry', 'trucking', 'lender',
                         'marketing', 'ENTERPRISES', 'game development', 'call-to-action', 'Manufacturing',
                         'software development', 'ads', 'Nanotechnology', 'manufacturers', 'hospitals',
                         'institutional-quality', 'chat assistants', 'inventory', 'secure solutions',
                         'board administration',
                         'software experts', 'clinical trials', 'manufacturer', 'open source', 'e-commerce solution',
                         'MARKETING', 'manufacturers', 'Service Providers', 'manufacturing',
                         'consortia', 'recruitment', 'EMR', 'asset management', 'build software', 'crops', 'Shopify',
                         'Big Commerce',
                         'DMARC', 'botnet', '(pDMP)', 'DMP', 'marketplaces', 'software systems', 'software solutions',
                         'Shopify', 'accounting firm'
                                    'inventory', 'DNS', 'prototyping', 'Commercial', 'medical providers', 'workplace',
                         'advertisers',
                         'accounting firms', 'biopharmaceutical', 'hospitals', 'Lawyers', 'logistics', 'delivery fleets',
                         'crops', 'content management', 'social media',
                         'publishers', 'property insurance', 'advertising software', 'public sector', 'marketing', 'lender',
                         'trucking', 'industry', 'government',
                         'marketing', 'platform', 'marketers', 'sales', 'digital marketing', 'dealerships',
                         'service providers',
                         'information services', 'cyber', 'security', 'charities', 'Infrastructure', 'employers', 'staff',
                         'leads', 'DevOps', 'open-source', 'Project Management', 'corporates', 'organizations',
                         'Asset Manager', 'managers', 'applicants', 'institutions', 'developers',
                         'teams', 'provider', 'restaurants', 'API', 'Enterprise', 'Teams', 'Agencies', 'Transportation',
                         'cities', 'teams', 'sellers', 'employers', 'operations software', 'brand', 'sales', 'teams',
                         'brand', 'consultancy', 'engineers', 'designers', 'healthcare providers', 'user experience',
                         'corporate', 'brand', 'service management', 'brand', 'oracle', 'teams', 'classroom', 'pubishers',
                         'broadcasters', 'organizations', 'sales', 'utilities', 'builders', 'teams',
                         'biotechnology', 'suppliers', 'sales', 'marketing', 'organizations', 'employees', 'doctors',
                         'employee']
            self.b2c_words = self.normalize_keywords(self.b2c_words)
            self.b2b_words = self.normalize_keywords(self.b2b_words)

            b2b_words_new, b2c_words_new, both_words_new = self.get_new_words()
            b2b_words_new, b2c_words_new, both_words_new = self.remove_common_words(b2b_words_new, b2c_words_new, both_words_new)

            # self.b2b_words = list(set(self.b2b_words + b2b_words_new))
            # self.b2c_words = list(set(self.b2c_words + b2c_words_new))
            self.b2b_words = b2b_words_new
            self.b2c_words = b2c_words_new
            self.b2b_b2c_words = both_words_new

            print(self.b2b_words)
            print(len(self.b2b_words))

            print(self.b2c_words)
            print(len(self.b2c_words))

            print(self.b2b_b2c_words)
            print(len(self.b2b_b2c_words))

        except Exception as e:
            print(e)

    def get_rules(self, files_header, filename_key):
        try:
            rules_df = DataManipulator.read_data(files_header, filename_key, self.logger, self.config_parser)
            return list(rules_df['Words'])
        except Exception as e:
            self.logger.error(e)

    def get_new_words(self):
        try:
            files_header = "filenames"
            filename_key = "b2b_words"
            b2b_words_new = self.get_rules(files_header, filename_key)
            filename_key = "b2c_words"
            b2c_words_new = self.get_rules(files_header, filename_key)
            filename_key = "both_words"
            both_words_new = self.get_rules(files_header, filename_key)

            return b2b_words_new, b2c_words_new, both_words_new
        except Exception as e:
            self.logger.error(e)

    def remove_common_words(self, b2b_words_new, b2c_words_new, both_words_new):
        b2b = b2b_words_new
        b2c = b2c_words_new

        b2b_words_new = list(set(b2b_words_new).difference(set(b2c_words_new)))
        b2b_words_new = list(set(b2b_words_new).difference(set(both_words_new)))

        b2c_words_new = list(set(b2c_words_new).difference(set(b2b)))
        b2c_words_new = list(set(b2c_words_new).difference(set(both_words_new)))

        both_words_new = list(set(both_words_new).difference(set(b2b)))
        both_words_new = list(set(both_words_new).difference(set(b2c)))

        return b2b_words_new, b2c_words_new, both_words_new

    def normalize_keywords(self, words_list):
        try:
            words_list = self.text_normalizer.text_preprocessing_list(words_list, self.logger)

            return words_list
        except Exception as e:
            self.logger(e)

    def normalize_descriptions(self, df_descriptions):
        try:
            df_descriptions = self.text_normalizer.text_preprocessing(df_descriptions, self.logger)

            return df_descriptions
        except Exception as e:
            self.logger.error(e)

    def classify(self, df, fields):
        try:

            predict_column = 'predictions'
            df[predict_column] = 0

            for index, row in df.iterrows():
                is_b2b = False
                is_b2c = False
                is_b2b_b2c = False

                description = row[fields[0]]
                description = description.split()
                prediction = 0

                # if any(word in description for word in self.b2b_b2c_words):
                #     print(index, 'Label 3 - Both', [word for word in self.b2b_b2c_words if word in description])
                #     is_b2b_b2c = True

                if any(word in description for word in self.b2c_words):
                    print(index, 'Label 2 - B2C', [word for word in self.b2c_words if word in description])
                    is_b2c = True
                if any(word in description for word in self.b2b_words):
                    print(index, 'Label 1 - B2B', [word for word in self.b2b_words if word in description])
                    is_b2b = True
                # if is_b2b and is_b2c:
                #     is_b2b_b2c = True

                if is_b2b_b2c:
                    prediction = 3
                elif is_b2b:
                    prediction = 1
                elif is_b2c:
                    prediction = 2

                df.set_value(index, predict_column, prediction)

            self.logger.info("Rule-based Classification Done")

            # df.to_csv("rule_based_output.csv", encoding='utf-8')
            return df[predict_column]

        except Exception as e:
            self.logger.error(e)


def main():
    rule_based_b2b_b2c = RuleBasedB2BB2C()
    logger = rule_based_b2b_b2c.logger
    config_parser = rule_based_b2b_b2c.config_parser

    FILE_SECTION = "filenames"
    filename_key = "new"
    fields = ["Company_Description", "Class_Label"]
    # filename_key = "unlabeled"
    # fields = ["CO:coDesc", "Fixed Final"]

    try:
        # Reading input data
        df = DataManipulator.read_data(FILE_SECTION, filename_key, logger, config_parser)

        # Filtering null rows
        df = DataManipulator.remove_nulldata(df[fields], logger)

        # Discarding rows with label X
        df = df[df[fields[1]] != 3]
        df = df[df[fields[1]] != 0]

        # Text preprocessing on descriptions
        df[fields[0]] = rule_based_b2b_b2c.normalize_descriptions(df[fields[0]])

        y_pred = rule_based_b2b_b2c.classify(df, fields)

        DataManipulator.evaluate(df[fields[1]], y_pred, logger)

        # Stats
        zero_rows = df[df['predictions'] == 0]
        print("Total Rows =", len(df))
        print("Rows classified by Rule-Based Algo: ", len(df) - len(zero_rows))
        print("Rows not classified by Rule-Based Algo: ", len(zero_rows))

    except Exception as e:
        logger.error(e)



if __name__ == "__main__":
    main()

