import json
import pandas as pd
import numpy as np
import gc
import datetime
import re
import seaborn as sns
import matplotlib.pyplot as plt

#######################################################################################
##                       Loading and Preprocessing Yelp Data                         ##
#######################################################################################
def get_business_data(file_name, state="AZ"):
    '''
    The function is designed to extract business data from the 
    given file name
    Inputs: 
        file_name: file name
        state: the specified state
    Return: dataframe for the business
    '''
    business = []

    with open(file_name, encoding="utf8") as fl:
        for i, line in enumerate(fl):
            business.append(json.loads(line))
            if i + 1 >= 100000:
                break

    df_business = pd.DataFrame(business)
    df_business = df_business[df_business["state"] == state]
    return df_business


def get_review_data(df_business, start_year, file_name, subset_n=5*(10**5)):
    '''
    The function is used to derive the reviews from the given file, using the
    businesses listed in the df_business. Start year and the total row size
    of review dataframe should also be specified.
    Inputs:
        df_business: dataframe for businesses
        start_year: starting year for rewiew data
        file_name: file name for review data
        subset_n: total row size searched in review dataframe
    '''
    reviews = []
    begin_date =  datetime.datetime(start_year, 1, 1, 0, 0)
    full_reviews = pd.DataFrame(columns=["business_id", "user_id", "stars", "date"])

    count = 0
    with open(file_name, encoding="utf8") as fl:
        for i, line in enumerate(fl):
            reviews.append(json.loads(line))
            count += 1
            if count + 1 >= 10000:
                df_reviews = pd.DataFrame(reviews)
                df_reviews['date'] = pd.to_datetime(df_reviews['date'])
                df_reviews = df_reviews[(df_reviews['date'] - begin_date) > datetime.timedelta(days = 0)]
                df_reviews = df_reviews[df_reviews["business_id"].isin(df_business["business_id"])]
                full_reviews = pd.concat([full_reviews, df_reviews[["business_id", "user_id", "stars", "date"]]], join="inner")
                df_reviews = ""
                count = 0
                gc.collect()
                print("currently, full_reviews contain: ", full_reviews.shape[0], " rows")
                if i + 1 > subset_n:
                    break
    
    return full_reviews


def get_final_matrix(full_reviews, review_threshold):
    '''
    The function is used to drop duplicated review and keep the latest one for a business-customer
    pair. A dataframe with customers and their corresponding rating (to all restaurant) will be returned.
    In order to control the sparsity of the matrix, only businesses who have review more than the review
    threshold will be included.
    Inputs: 
        full_review: review dataframe
        review_threshold: the minimum review for a customer
    Return: matrix for every business-customer pair
    '''
    final_matrix = full_reviews.sort_values(by="date", ascending=False).drop_duplicates(
                           subset=["business_id","user_id"], keep='first', inplace=False)
    final_matrix = final_matrix[["business_id","user_id", "stars"]].pivot(
                           index='business_id', columns='user_id', values='stars')
    return final_matrix[final_matrix.count(axis = 1) > review_threshold]


#######################################################################
##                             Imputer                               ##
#######################################################################
def initial_imputer(df, indicator_matrix=True):
    '''
    The function is used to impute the missing value in the customer-business
    matrix and generate a complete dataframe. The missing value of each row will
    be replaced by the random draw from the normal distribution (which is assumed)
    of reviews for each business.
    Inputs:
        df: business-customer matrix
    Return: complete matrix, matrix of missing indicator
    '''
    for idx in np.arange(0, df.shape[0], 1):
        mu = df[idx, :][~np.isnan(df[idx, :])].mean()
        sigma = df[idx, :][~np.isnan(df[idx, :])].std()
        null_num = len(df[idx,:][np.isnan(df[idx, :])])
        imputed_vals = np.random.normal(mu, sigma, null_num)
        df[idx,:][np.isnan(df[idx, :])] = imputed_vals
    
    if indicator_matrix:
        ind_matrix = np.isnan(df)
        return df, ind_matrix
    
    return df


########################################################################
##                        Finding Patterns in labels                  ##
########################################################################
def find_re(txt, pattern):
    if re.findall(pattern, txt) != []:
        return True
    return False