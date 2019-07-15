"""
    Author: Hsuan-Hau Liu
    Date:   June, 18th, 2019
    Description: Implementation of various collaborating filtering algorithms.
"""


from recommendation_system import collaborative_filtering as cf
from recommendation_system import util


# PySpark import
from pyspark import SparkContext


def main():
    """ Main function """
    context = SparkContext('local', 'CF')
    context.setLogLevel('OFF')
    train_file, test_file, case_num, output_file = util.parse_inputs()

    train_data = util.parse_csv(context.textFile(train_file), header=True)
    test_data = util.parse_csv(context.textFile(test_file), header=True)

    # make predictions
    predictions = cf.predict_rating(train_data, test_data, case_num)
    util.output_csv(output_file, predictions)


if __name__ == '__main__':
    main()