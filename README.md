# yelp-recommendation-system
Build a Yelp business recommendation system using various [collaborative-filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) technique with the [Yelp dataset](https://www.yelp.com/dataset/challenge).

Collaborative-filtering algorithms are methods used to build recommendation systems (Amazon item recommendation, Youtube video recommendation, etc.). There are many different recommendation algorithms, and this exercise includes three, which are content-based CF, user-based CF, and ALS algorithm. The goal of these algorithms are to predict missing ratings of a user-business pair (i.e. user gives a 3-star rating on a restaurant). Then, we can use these results to recommend businesses that users have not rated yet but are likely to rate highly.

In this exercise, a train file and a validation file are given. We want to use the data in train file to predict each user-business pair in the validation file. The outputs are files containing predicted ratings for each pair in CSV format. Inputs are sample outputs are included in their corresponding folders.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python >= 3.6
- Spark >= 2.3.3

### Installation

Download the project by cloning the repository.

```
git clone https://github.com/hsuanhauliu/yelp-recommendation-system.git
```

### Usage

Follow the command below to run the program. The case number indicates which algorithm we want to use. 1 for user-based CF, 2 for item-based, and 3 for ALS.

```
python3 main.py [training_file] [test_file] [case_number] [output_file]
```

Example:
```
python3 main.py inputs/yelp_train.csv inputs/yelp_val.csv 1 output1.csv
```
