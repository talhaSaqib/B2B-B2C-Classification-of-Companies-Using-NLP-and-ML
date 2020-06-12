"""

This file contains a class that configures basic settings needed for every Python program, such as:

> Config Parser
> Logger
> Panda Display

"""

__author__ = "Talha Saqib"

# Local Imports
import logging

# Third-party Imports
import pandas as pd
import configparser as config
import matplotlib.pyplot as plt


class Configurator(object):

    def __init__(self):
        pass

    @staticmethod
    def get_config_parser():
        config_parser = config.ConfigParser()
        config_parser.read("config.ini")
        return config_parser

    @staticmethod
    def get_logger():
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s-%(levelname)s-%(funcName)s-%(lineno)d-%(message)s')
        logger = logging.getLogger()
        return logger

    @staticmethod
    def set_warnings_off():
        import warnings
        warnings.filterwarnings("ignore")

    @staticmethod
    def set_pandas_display():
        # Overriding output display dimensions
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.max_rows', 7300)

    @staticmethod
    def set_plot():
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 6
        fig_size[1] = 6
        plt.rcParams["figure.figsize"] = fig_size

    def set_configurator(self):
        config_parser = self.get_config_parser()
        logger = self.get_logger()
        self.set_pandas_display()
        self.set_plot()
        self.set_warnings_off()
        return config_parser, logger






