import kagglehub
from kagglehub import KaggleDatasetAdapter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf


def load_kaggle_data(file_name, data_handle):
  """
  Loads Kaggle data into a Pandas dataframe.

  Args:
  - file_name: file with .csv extenstion
  - data_handle: handle found on Kaggle site

  Returns:
  - df: Pandas dataframe with data loaded from Kaggle
  """
  # Load the latest version
  df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    data_handle,
    file_name
  )

  return df

def binary_scatter(df, x_col, y_col, target_col):
  """
  Plot some data columns in the dataframe with target col
  to make scatters different colors

  Args:
  - df (pd.Dataframe): Pandas dataframe for binary logistic regression
  - x_col (str): x column to use
  - y_col (str): y column to use
  - target_col (int): target column with 1s or 0s

  Returns:
  - None
    Plots a scatter plot 
  """
  sns.scatterplot(
      data=df,
      x=x_col,
      y=y_col,
      hue=target_col,
      style=target_col,
      palette={1: 'r', 0: 'b'},
      markers={1: 'X', 0: 'o'}
  )

  plt.xlabel(f'{x_col}')
  plt.ylabel(f'{y_col}')
  plt.legend(title=f'{target_col}')
  plt.show()

def binary_3D_plot(df, x_col, y_col, z_col, target_col, title=None, size_col=None):
  """
  Plot a 3D scatter plot with dataframe columns and color the points
  red or blue based on the target column.

  Args:
  - df (pd.Dataframe): Pandas dataframe for training
  - x_col (str): x column of df to be plotted
  - y_col (str): y column of df to be plotted
  - z_col (str): z column of df to be plotted
  - target_col (str): target column to use for coloring points
  - title (str): title of the plot
  - size_col (str): column to use to change size of points (optional)

  Returns:
  - None : plots a 3D figure
  """
  # Make a categorical target variable for plotting
  df[f'{target_col}_cat'] = df[f'{target_col}'].astype(str)

  fig = px.scatter_3d(
      df,
      x=x_col,
      y=y_col,
      z=z_col,
      color=f'{target_col}_cat',                          
      size=size_col,             
      color_discrete_map={
          '1': 'red',  
          '0': 'blue'   
      },
      title=title,
      opacity=0.8,
  )

  fig.update_layout(
      scene=dict(
          xaxis_title=f'{x_col}',
          yaxis_title=f'{y_col}',
          zaxis_title=f'{z_col}',
      ),
      legend_title_text=f'{target_col}'
  )

  fig.show()

def split_dataset(X, y):
  """
  Splits the dataset into train (60%), cv (20%), and test sets (20%)

  Args:
  - X (np.array): array of features (m, n)
  - y (np.array): array of targets (m,)

  Returns:
  - x_train (np.array): train dataset
  - y_train (np.array): training targets
  - x_cv (np.array): cross-validation dataset
  - y_cv (np.array): cross-validation targets
  - x_test (np.array): test dataset
  - y_test (np.array): test targets
  """
  # Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
  x_train, x_, y_train, y_ = train_test_split(X, y, test_size=0.40, random_state=1)

  # Split the 40% subset above into two: one half for cross validation and the other for the test set
  x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

  # Delete temporary variables
  del x_, y_

  return x_train, y_train, x_cv, y_cv, x_test, y_test

def determine_prediction(model_predict, threshold = 0.5):
    """
    Determine whether the model predicts 1 or 0 based on threshold.
    
    Args:
    - model_predict: model prediction of sigmoid function
    - threshold (default 0.5): above the threshold value, yhat = 1
    
    Returns:
    - yhat: list of 1s and 0s 
    """
    yhat = []
    for val in model_predict:
        if val >= threshold:
            yhat.append(1)
        else:
            yhat.append(0)
    yhat = np.array(yhat)
    return yhat

def lambda_analysis(models, lambdas, x_train, x_cv, y_train, y_cv):
    """
    Plots the MSE of the train and CV data of a model with different values of lambdas

    Args:
    - models: list of models trained with different lambdas
    - lambdas: list of lambdas used to train models
    - x_train: training set
    - x_cv: cross-validation set
    - y_train: training targets
    - y_cv: cross-validation targets

    Returns:
    - plot of lambda vs MSE
    """
    mse_train = []
    mse_cv = []
    for i in range(len(lambdas)):
        # Calculate predictions for train and CV datasets
        train_predict = tf.nn.sigmoid(models[i].predict(x_train)).numpy()[0]
        cv_predict = tf.nn.sigmoid(models[i].predict(x_cv)).numpy()[0]

        # Calculate binary prediction for train and CV datasets
        yhat_train = determine_prediction(train_predict)
        yhat_cv = determine_prediction(cv_predict)

        # Calculate mse for train and CV datasets
        mse_train.append(mean_squared_error(y_train, yhat_train))
        mse_cv.append(mean_squared_error(y_cv, yhat_cv))
    
    plt.plot(lambdas, mse_train, linewidth=3, label='Train MSE')
    plt.plot(lambdas, mse_cv, linewidth=3, label='CV MSE')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

def plot_correlation(df):
    """
    Check the correlation of features with a heatmap.
    
    Args:
    - df: pandas.Dataframe

    Returns:
    - Heatmap of correlations between each column
    """
    corr_df = df.corr()
    plt.figure(figsize = (10, 6), dpi = 84)
    sns.heatmap(corr_df, annot = True)
    plt.show()

def plot_null_values(df):
    """
    Plot null value percentages in the dataframe
    
    Args:
    - df: pandas.Dataframe

    Returns:
    - None: plots the null value percentages in a bar chart
    """
    data_nulls = (df.apply(lambda x: x.isnull().value_counts()).T[True]/len(df)*100).reset_index(name='count')
    
    fig = plt.figure(figsize=(12,6))
    fig = sns.barplot(data_nulls, x="index", y="count")
    fig.set_title('Null Values in the Data', fontsize=30)
    fig.set_xlabel('features', fontsize=12)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=45)
    fig.set_ylabel('% of null values', fontsize=12)
    fig.bar_label(fig.containers[0], fmt='%.1f')
    plt.tight_layout()
    plt.show()