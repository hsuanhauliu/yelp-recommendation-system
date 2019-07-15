""" Utility functions. """


from argparse import ArgumentParser


def parse_inputs():
    """ Parser function to take care of the inputs """
    parser = ArgumentParser(description='Input input filename and output filename')
    parser.add_argument('train_file_name', type=str,
                        help='Enter the path of the training file.')
    parser.add_argument('test_file_name', type=str,
                        help='Enter the path of the testing file.')
    parser.add_argument('case_number', type=int,
                        choices=[1, 2, 3],
                        help='Enter case number.')
    parser.add_argument('output_file_name', type=str,
                        help='Enter the path of the testing file.')
    args = parser.parse_args()

    return (args.train_file_name, args.test_file_name,
            args.case_number, args.output_file_name)


def parse_csv(data, header=False):
    """ Parse CSV data in RDD """
    data = data.map(lambda x: x.split(','))
    if header:
        header = data.first()
        data = data.filter(lambda x: x != header)\
               .map(lambda x: (x[0], x[1], float(x[2])))
    return data


def output_csv(filename, data):
    """ Save the output to a file """
    with open(filename, 'w') as write_file:
        write_file.write('user_id, business_id, prediction\n')
        for user, business, rating in data.toLocalIterator():
            write_file.write(','.join([user, business, str(rating)]))
            write_file.write('\n')
