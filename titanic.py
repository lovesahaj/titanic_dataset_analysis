from datetime import datetime

intial_time = datetime.now()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
print(dftrain.head())

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMN = ['sex', 'n_siblings_spouses','parch','class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMN = ['age', 'fare']

feature_column = []

for feature_name in CATEGORICAL_COLUMN:
  vocabulary = dftrain[feature_name].unique()
  feature_column.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMN:
  feature_column.append(tf.feature_column.numeric_column(feature_name, dtype= tf.float32))


def make_input_fn(data_df, label_df, num_epochs=10, shuffle = True, batch_size = 32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs = 1, shuffle = False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_column)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)
time_elapsed = datetime.now() - intial_time


print("The accuracy was", result['accuracy'], 'with time elapsed', time_elapsed)