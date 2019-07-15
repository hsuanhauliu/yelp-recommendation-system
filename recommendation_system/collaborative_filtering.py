""" Collaborative Filtering algorithms implementations. """


from math import sqrt

# PySpark import
from pyspark.mllib.recommendation import ALS, Rating


def build_rating_table(data):
    """ Build rating look-up table for user and business pair """
    table = {}
    for user, business, rating in data.toLocalIterator():
        table[(user, business)] = rating
    return table


def build_user_sets(data):
    """ Aggregate all businesses to users who have rated them """
    user_rdd = data.map(lambda x: (x[0], x[1])).groupByKey()
    user_sets = {}
    for key, vals in user_rdd.toLocalIterator():
        user_sets[key] = set(vals)
    return user_sets


def build_business_sets(data):
    """ Aggregate all users to businesses who have been rated by them """
    business_rdd = data.map(lambda x: (x[1], x[0])).groupByKey()
    business_sets = {}
    for key, vals in business_rdd.toLocalIterator():
        business_sets[key] = set(vals)
    return business_sets


def find_median(attri_sets, ratings, case):
    """ Find the median rating from the set """
    ave = []
    if case == 'user':
        for user, businesses in attri_sets.items():
            ave.append(sum(ratings[(user, b)] for b in businesses) / len(businesses))
    elif case == 'business':
        for business, users in attri_sets.items():
            ave.append(sum(ratings[(u, business)] for u in users) / len(users))
    ave.sort()
    return ave[len(ave) // 2]


def predict_rating(train_data, test_data, case_num):
    """ Predict user, business rating """
    rating_table = build_rating_table(train_data)
    user_sets = build_user_sets(train_data)
    median_user_rating = find_median(user_sets, rating_table, 'user')
    if case_num == 1:
        return predict_rating_model_based(train_data, test_data, median_user_rating)

    weights = {}
    averages = {}
    business_sets = build_business_sets(train_data)
    median_business_rating = find_median(business_sets, rating_table, 'business')

    if case_num == 2:
        return test_data.map(lambda x: predict_rating_user_based(x[0], x[1],
                             rating_table, user_sets, business_sets, weights,
                             averages, median_user_rating, median_business_rating,
                             case_num))

    return test_data.map(lambda x: predict_rating_item_based(x[0], x[1],
                         rating_table, user_sets, business_sets, weights, averages,
                         median_user_rating, median_business_rating, case_num))


def predict_rating_model_based(train_data, test_data, median_rating):
    """ Predict rating for user, business pair using model-based CF """
    user_mapping, business_mapping = _create_mapping(train_data, test_data)
    train_data = train_data.map(lambda x: Rating(user_mapping[x[0]], business_mapping[x[1]], x[2]))
    test_data = test_data.map(lambda x: ((x[0], x[1]), 1))
    mapped_test_data = test_data.map(lambda x: (user_mapping[x[0][0]], business_mapping[x[0][1]]))

    # create model
    model = ALS.train(train_data, 5, 10)
    predictions = model.predictAll(mapped_test_data)\
                  .map(lambda x: ((user_mapping[x[0]], business_mapping[x[1]]), x[2]))

    # collect all the cold start
    cold_starts = test_data.subtractByKey(predictions).map(lambda x: (x[0], median_rating))

    return predictions.union(cold_starts)\
           .map(lambda x: (x[0][0], x[0][1], _round_rating_model_based(x[1])))


def _create_mapping(train_data, test_data):
    """ Create two-way mapping between ID and indices """
    user_mapping = {}
    business_mapping = {}
    user_counter = business_counter = 0
    for user, business, _ in train_data.toLocalIterator():
        if user not in user_mapping:
            user_mapping[user] = user_counter
            user_mapping[user_counter] = user
            user_counter += 1
        if business not in business_mapping:
            business_mapping[business] = business_counter
            business_mapping[business_counter] = business
            business_counter += 1

    for user, business, _ in test_data.toLocalIterator():
        if user not in user_mapping:
            user_mapping[user] = user_counter
            user_mapping[user_counter] = user
            user_counter += 1
        if business not in business_mapping:
            business_mapping[business] = business_counter
            business_mapping[business_counter] = business
            business_counter += 1

    return user_mapping, business_mapping


def predict_rating_user_based(user, business, ratings, user_sets, business_sets,
                              user_weights, user_averages, median_user_rating,
                              median_business_rating, case_num):
    """ Predict rating for user, business pair using user-based CF """
    if (user, business) in ratings:
        return (user, business, ratings[(user, business)])

    # cold start for new business
    if business not in business_sets:
        return (user, business, median_business_rating)

    # cold start for new user
    if user not in user_sets:
        return (user, business, median_user_rating)

    weights = _get_neighbor_weights(user, business_sets[business], ratings,
                                   user_sets, user_weights, case_num)
    ave_rating = _get_ave_rating(user, ratings, user_sets, user_averages)
    weighted_rating = _get_weighted_rating(business, weights, ratings, user_sets, user_averages)
    return (user, business, _round_rating_user_based(ave_rating + weighted_rating))


def predict_rating_item_based(user, business, ratings, user_sets, business_sets,
                              business_weights, business_averages, median_user_rating,
                              median_business_rating, case_num):
    """ Predict rating for user, business pair using item-based CF """
    if (user, business) in ratings:
        return (user, business, ratings[(user, business)])

    # cold start for new business
    if business not in business_sets:
        return (user, business, median_business_rating)

    # cold start for new user
    if user not in user_sets:
        return (user, business, median_user_rating)

    weights = _get_neighbor_weights(business, user_sets[user], ratings,
                                   business_sets, business_weights, case_num)

    total_weights = weighted_rating = 0.0
    if weights:
        for neighbor, weight in weights:
            total_weights += abs(weight)
            weighted_rating += ratings[(user, neighbor)] * weight
        weighted_rating = weighted_rating / total_weights
    else:
        if business not in business_averages:
            all_users = business_sets[business]
            average_rating = sum(ratings[(u, business)] for u in all_users) / len(all_users)
            business_averages[business] = average_rating
        weighted_rating = business_averages[business]

    return (user, business, _round_rating_item_based(weighted_rating))


def _get_neighbor_weights(target, neighbors, ratings, attri_sets, prev_weights,
                         case_num):
    """ Calculate weight between target and neighbor """
    all_weights = []
    for neighbor in neighbors:
        pair = (target, neighbor)
        weight = 0.0
        if pair in prev_weights:
            weight = prev_weights[pair]
        else:
            weight = _get_weight(target, neighbor, ratings, attri_sets, case_num)
            prev_weights[pair] = weight

    return all_weights


def _get_weight(target, neighbor, ratings, attri_sets, case_num):
    """ Calculate weight for both users """
    co_rated = attri_sets[target].intersection(attri_sets[neighbor])
    vec_1 = []
    vec_2 = []
    if case_num == 2:
        vec_1 = [ratings[(target, attri)] for attri in co_rated]
        vec_2 = [ratings[(neighbor, attri)] for attri in co_rated]
    elif case_num == 3:
        vec_1 = [ratings[(attri, target)] for attri in co_rated]
        vec_2 = [ratings[(attri, neighbor)] for attri in co_rated]
    else:
        raise ValueError("Wrong case number")

    weight = 0.0
    num = len(co_rated)
    if num > 1:
        # find average
        ave_1 = sum(vec_1) / num
        ave_2 = sum(vec_2) / num

        # normalize vector
        vec_1 = [n - ave_1 for n in vec_1]
        vec_2 = [n - ave_2 for n in vec_2]
        numerator = sum(vec_1[i] * vec_2[i] for i in range(num))

        # numerator is 0
        if not numerator:
            return 0.0

        # calculate magnitude of both vectors
        denominator = sqrt(sum(n * n for n in vec_1) * sum(n * n for n in vec_2))
        weight = numerator / denominator

    return weight


def _get_ave_rating(target, ratings, user_sets, averages):
    """ Calculate target average rating """
    if target not in averages:
        attri = user_sets[target]
        sum_ratings = sum(ratings[(target, a)] for a in attri)
        n_ratings = len(attri)
        averages[target] = (sum_ratings, n_ratings)
    return averages[target][0] / averages[target][1]


def _get_weighted_rating(business, weights, ratings, user_sets, user_averages):
    """ Calculate the weighted rating of neighbors """
    # if no similar neighbors
    if not weights:
        return 0.0

    # find weighted rating of each neighbor
    total_weights = weighted_rating = 0.0
    for neighbor, weight in weights:
        if neighbor not in user_averages:
            neighbor_b = user_sets[neighbor]
            sum_ratings = sum(ratings[(neighbor, b)] for b in neighbor_b)
            n_ratings = len(neighbor_b)
            user_averages[neighbor] = (sum_ratings, n_ratings)
        neighbor_ave_rating = (user_averages[neighbor][0] - ratings[(neighbor, business)]) / (user_averages[neighbor][1] - 1)

        weighted_rating += (ratings[(neighbor, business)] - neighbor_ave_rating) * weight
        total_weights += abs(weight)

    return weighted_rating / total_weights


def _round_rating_model_based(rating):
    """ Round rating to reasonable value """
    if rating > 5.0:
        return 5.0

    if rating < 2.0:
        return 2.0

    return rating


def _round_rating_user_based(rating):
    """ Round rating to reasonable value """
    if rating > 4.5:
        return 4.5

    if rating < 2.0:
        return 2.0

    return rating


def _round_rating_item_based(rating):
    """ Round rating to reasonable value """
    if rating > 4.0:
        return 4.0

    if rating < 2.0:
        return 2.0

    return rating
