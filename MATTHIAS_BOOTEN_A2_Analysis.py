#!/usr/bin/env python
# coding: utf-8

# <h3>Read in the dataset.</h3><br>
# Importing appropriate libraries, setting Panda print options and reading in the file as a dataset.

# In[1]:


# timeit

# Student Name : Matthias Booten
# Cohort       : 5, Valencia

# Import libraries

# Standard essential libraries

import pandas                  as pd                      # data science essentials
import matplotlib.pyplot       as plt                     # essential graphical output
import seaborn                 as sns                     # enhanced graphical outputimport pandas as pd
import statsmodels.formula.api as smf                     # regression modeling

# Train test split libraries

from   sklearn.model_selection import train_test_split    # train/test split

# KNeigbors libraries

from sklearn.neighbors import KNeighborsClassifier        # KNN for Classification
from   sklearn.preprocessing   import StandardScaler      # standard scaler

# Logistic regression and confusion matrix libraries

from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.metrics import confusion_matrix         # confusion matrix
from sklearn.metrics import roc_auc_score            # auc score

# Gradient Boosting classifier library

from sklearn.ensemble import GradientBoostingClassifier

# CART model packages

from sklearn.tree import DecisionTreeClassifier      # classification trees

# Confusion matrix

from sklearn.metrics import confusion_matrix   

# libraries for classification trees
from sklearn.tree import DecisionTreeClassifier      # classification trees
from sklearn.tree import export_graphviz             # exports graphics
from sklearn.externals.six import StringIO           # saves objects in memory
from IPython.display import Image                    # displays on frontend
import pydotplus                                     # interprets dot objects

# Set pandas print options

pd.set_option('display.max_rows'   , 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width'      , 1000)

# Specify file name
original_df = 'Apprentice_Chef_Dataset.xlsx'


# Read the file into Python
chefdf = pd.read_excel(original_df)










#########################
# mv_flagger
#########################

def mv_flagger(df):
    """
Flags all columns that have missing values with 'm-COLUMN_NAME'.

PARAMETERS
----------
df : DataFrame to flag missing values is called chefdf


RETURNS
-------
DataFrame with missing value flags."""


    for col in df:

        if df[col].isnull().astype(int).sum() > 0:
            df['m_'+col] = df[col].isnull().astype(int)
            
    return df



# text split of missing names

def text_split_names(col, df, sep=' ', new_col_name='number_of_names'):
    """
Splits values in a string Series (as part of a DataFrame) and sums the number
of resulting items. Automatically appends summed column to original DataFrame.

PARAMETERS
----------
col          : column to split
df           : DataFrame where column is located
sep          : string sequence to split by, default ' '
new_col_name : name of new column after summing split, default
               'number_of_names'
"""
    
    df[new_col_name] = 0
    
    
    for index, val in df.iterrows():
        df.loc[index, new_col_name] = len(df.loc[index, col].split(sep = ' '))
        
#counting strings in names
text_split_names(col = 'NAME',
                   df  = chefdf)






chefdf = mv_flagger(chefdf)
chefdf.head(5)



# Dropping variables 
chefdf = chefdf.drop(['FIRST_NAME', 'FAMILY_NAME','m_FAMILY_NAME', 'NAME'], axis = 1)




####################
#FEATURE ENGINEERING
####################

#REVENUE
REVENUE_hi = 4500

#TOTAL_MEALS_ORDERED
TOTAL_MEALS_ORDERED_lo = 30
TOTAL_MEALS_ORDERED_hi = 200

#UNIQUE_MEALS_PURCH
UNIQUE_MEALS_PURCH_lo = 1
UNIQUE_MEALS_PURCH_hi = 9

#CONTACTS_W_CUSTOMER_SERVICE
CONTACTS_W_CUSTOMER_SERVICE_lo = 4.0
CONTACTS_W_CUSTOMER_SERVICE_hi = 8.0

#AVG_TIME_PER_SITE_VISIT
AVG_TIME_PER_SITE_VISIT_lo = 0
AVG_TIME_PER_SITE_VISIT_hi = 200

#WEEKLY_PLAN
WEEKLY_PLAN_lo = 0
WEEKLY_PLAN_hi = 15

#EARLY_DELIVERIES
EARLY_DELIVERIES_lo = 0
EARLY_DELIVERIES_hi = 4 

#LATE_DELIVERIES
LATE_DELIVERIES_lo = 0
LATE_DELIVERIES_hi = 6

#AVG_PREP_VID_TIME
AVG_PREP_VID_TIME_lo = 100
AVG_PREP_VID_TIME_hi = 200

#LARGEST_ORDER_SIZE
LARGEST_ORDER_SIZE_lo = 2
LARGEST_ORDER_SIZE_hi = 6

#MEDIAN_MEAL_RATING
MEDIAN_MEAL_RATING_lo = 2
MEDIAN_MEAL_RATING_hi = 4

#AVG_CLICKS_PER_VISIT
AVG_CLICKS_PER_VISIT_lo = 11
AVG_CLICKS_PER_VISIT_hi = 17

##############################################################################
## Feature Engineering (outlier thresholds)                                 ##
##############################################################################

# Developing features (columns) for outliers

# REVENUE

chefdf['out_REVENUE'] = 0 
condition_hi = chefdf.loc[0:,'out_REVENUE'][chefdf['REVENUE'] > REVENUE_hi]

chefdf['out_REVENUE'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

# TOTAL_MEALS_ORDERED

chefdf['out_TOTAL_MEALS_ORDERED'] = 0
condition_hi = chefdf.loc[0:,'out_TOTAL_MEALS_ORDERED'][chefdf['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_hi]
condition_lo = chefdf.loc[0:,'out_TOTAL_MEALS_ORDERED'][chefdf['TOTAL_MEALS_ORDERED'] < TOTAL_MEALS_ORDERED_lo]

chefdf['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# UNIQUE_MEALS_PURCH

chefdf['out_UNIQUE_MEALS_PURCH'] = 0
condition_hi = chefdf.loc[0:,'out_UNIQUE_MEALS_PURCH'][chefdf['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_hi]
condition_lo = chefdf.loc[0:,'out_UNIQUE_MEALS_PURCH'][chefdf['UNIQUE_MEALS_PURCH'] < UNIQUE_MEALS_PURCH_lo]

chefdf['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE

chefdf['out_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi = chefdf.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][chefdf['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_hi]
condition_lo = chefdf.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][chefdf['CONTACTS_W_CUSTOMER_SERVICE'] < CONTACTS_W_CUSTOMER_SERVICE_lo]

chefdf['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)


# AVG_TIME_PER_SITE_VISIT

chefdf['out_AVG_TIME_PER_SITE_VISIT'] = 0
condition_hi = chefdf.loc[0:,'out_AVG_TIME_PER_SITE_VISIT'][chefdf['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_hi]
condition_lo = chefdf.loc[0:,'out_AVG_TIME_PER_SITE_VISIT'][chefdf['AVG_TIME_PER_SITE_VISIT'] < AVG_TIME_PER_SITE_VISIT_lo]

chefdf['out_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# WEEKLY_PLAN

chefdf['out_WEEKLY_PLAN'] = 0
condition_hi = chefdf.loc[0:,'out_WEEKLY_PLAN'][chefdf['WEEKLY_PLAN'] > WEEKLY_PLAN_hi]
condition_lo = chefdf.loc[0:,'out_WEEKLY_PLAN'][chefdf['WEEKLY_PLAN'] < WEEKLY_PLAN_lo]

chefdf['out_WEEKLY_PLAN'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_WEEKLY_PLAN'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# EARLY_DELIVERIES
chefdf['out_EARLY_DELIVERIES'] = 0
condition_hi = chefdf.loc[0:,'out_EARLY_DELIVERIES'][chefdf['EARLY_DELIVERIES'] > EARLY_DELIVERIES_hi]
condition_lo = chefdf.loc[0:,'out_EARLY_DELIVERIES'][chefdf['EARLY_DELIVERIES'] < EARLY_DELIVERIES_lo]

chefdf['out_EARLY_DELIVERIES'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_EARLY_DELIVERIES'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# LATE_DELIVERIES

chefdf['out_LATE_DELIVERIES'] = 0
condition_hi = chefdf.loc[0:,'out_LATE_DELIVERIES'][chefdf['LATE_DELIVERIES'] > LATE_DELIVERIES_hi]
condition_lo = chefdf.loc[0:,'out_LATE_DELIVERIES'][chefdf['LATE_DELIVERIES'] < LATE_DELIVERIES_lo]

chefdf['out_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_LATE_DELIVERIES'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# AVG_PREP_VID_TIME

chefdf['out_AVG_PREP_VID_TIME'] = 0
condition_hi = chefdf.loc[0:,'out_AVG_PREP_VID_TIME'][chefdf['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_hi]
condition_lo = chefdf.loc[0:,'out_AVG_PREP_VID_TIME'][chefdf['AVG_PREP_VID_TIME'] < AVG_PREP_VID_TIME_lo]

chefdf['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# LARGEST_ORDER_SIZE

chefdf['out_LARGEST_ORDER_SIZE'] = 0
condition_hi = chefdf.loc[0:,'out_LARGEST_ORDER_SIZE'][chefdf['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_hi]
condition_lo = chefdf.loc[0:,'out_LARGEST_ORDER_SIZE'][chefdf['LARGEST_ORDER_SIZE'] < LARGEST_ORDER_SIZE_lo]

chefdf['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# MEDIAN_MEAL_RATING

chefdf['out_MEDIAN_MEAL_RATING'] = 0
condition_hi = chefdf.loc[0:,'out_MEDIAN_MEAL_RATING'][chefdf['MEDIAN_MEAL_RATING'] > MEDIAN_MEAL_RATING_hi]
condition_lo = chefdf.loc[0:,'out_MEDIAN_MEAL_RATING'][chefdf['MEDIAN_MEAL_RATING'] < MEDIAN_MEAL_RATING_lo]

chefdf['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# AVG_CLICKS_PER_VISIT

chefdf['out_AVG_CLICKS_PER_VISIT'] = 0
condition_hi = chefdf.loc[0:,'out_AVG_CLICKS_PER_VISIT'][chefdf['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_hi]
condition_lo = chefdf.loc[0:,'out_AVG_CLICKS_PER_VISIT'][chefdf['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_lo]

chefdf['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

###################
#TREND ANALYSIS
###################

# Setting trend-based thresholds

change_TOTAL_MEALS_ORDERED_hi            = 250 #Data scatters above this value
change_UNIQUE_MEALS_PURCH_hi             = 9   #Data scatters above this value
change_TOTAL_PHOTOS_VIEWED_hi            = 500 #Data scatters above this value
change_CONTACTS_W_CUSTOMER_SERVICE_hi    = 10  #Start of a downward trend then trend stops to flat line
change_AVG_TIME_PER_SITE_VISIT_hi        = 300 #Data scatters above this value 
change_CANCELLATIONS_BEFORE_NOON_hi      = 8   #Data scatters above this value
change_LATE_DELIVERIES_hi                = 10  #Data scatters above this value
change_AVG_PREP_VID_TIME_hi              = 290 #Data scatters above this value
change_LARGEST_ORDER_SIZE_hi             = 9   #Data scatters above this value
change_AVG_CLICKS_PER_VISIT_hi           = 10  #Starts from 8 to then and then has downward trend

# Change takes place at

change_MOBILE_NUMBER_at                  = 1 # According to graph it has more points present in higher revenue ranges for value = 1
change_TOTAL_PHOTOS_VIEWED_at            = 0 #strong concentration
change_WEEKLY_PLAN_at                    = 0 #High density around zero
change_TOTAL_PHOTOS_VIEWED_at            = 0 #heavy concentration
change_MEDIAN_MEAL_RATING_at             = 4 #discovered through categorical var analysis
change_UNIQUE_MEALS_PURCH_at             = 1 #strong concentration at 1 with some very high values for revenue
change_CANCELLATIONS_AFTER_NOON_at       = 0 #strongly zero inflated with some higher revenue values around zero

# Trend-based feature template

# change_TOTAL_MEALS_ORDERED_hi

chefdf['change_TOTAL_MEALS_ORDERED_hi'] = 0
condition = chefdf.loc[0:,'change_TOTAL_MEALS_ORDERED_hi'][chefdf['TOTAL_MEALS_ORDERED'] > change_TOTAL_MEALS_ORDERED_hi]

chefdf['change_TOTAL_MEALS_ORDERED_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# UNIQUE_MEALS_PURCH_hi

chefdf['change_UNIQUE_MEALS_PURCH_hi'] = 0
condition = chefdf.loc[0:,'change_UNIQUE_MEALS_PURCH_hi'][chefdf['UNIQUE_MEALS_PURCH'] > change_UNIQUE_MEALS_PURCH_hi]

chefdf['change_UNIQUE_MEALS_PURCH_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# change_TOTAL_PHOTOS_VIEWED_hi

chefdf['change_TOTAL_PHOTOS_VIEWED_hi'] = 0
condition = chefdf.loc[0:,'change_TOTAL_PHOTOS_VIEWED_hi'][chefdf['TOTAL_PHOTOS_VIEWED'] > change_TOTAL_PHOTOS_VIEWED_hi]

chefdf['change_TOTAL_PHOTOS_VIEWED_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# change_CONTACTS_W_CUSTOMER_SERVICE_hi

chefdf['change_CONTACTS_W_CUSTOMER_SERVICE_hi'] = 0
condition = chefdf.loc[0:,'change_CONTACTS_W_CUSTOMER_SERVICE_hi'][chefdf['CONTACTS_W_CUSTOMER_SERVICE'] > change_CONTACTS_W_CUSTOMER_SERVICE_hi]

chefdf['change_CONTACTS_W_CUSTOMER_SERVICE_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)
# change_AVG_TIME_PER_SITE_VISIT_hi

chefdf['change_AVG_TIME_PER_SITE_VISIT_hi'] = 0
condition = chefdf.loc[0:,'change_AVG_TIME_PER_SITE_VISIT_hi'][chefdf['AVG_TIME_PER_SITE_VISIT'] > change_AVG_TIME_PER_SITE_VISIT_hi]

chefdf['change_AVG_TIME_PER_SITE_VISIT_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# change_CANCELLATIONS_BEFORE_NOON_hi

chefdf['change_CANCELLATIONS_BEFORE_NOON_hi'] = 0
condition = chefdf.loc[0:,'change_CANCELLATIONS_BEFORE_NOON_hi'][chefdf['CANCELLATIONS_BEFORE_NOON'] > change_CANCELLATIONS_BEFORE_NOON_hi]

chefdf['change_CANCELLATIONS_BEFORE_NOON_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# change_LATE_DELIVERIES_hi

chefdf['change_LATE_DELIVERIES_hi'] = 0
condition = chefdf.loc[0:,'change_LATE_DELIVERIES_hi'][chefdf['LATE_DELIVERIES'] > change_LATE_DELIVERIES_hi]

chefdf['change_LATE_DELIVERIES_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# change_AVG_PREP_VID_TIME_hi

chefdf['change_AVG_PREP_VID_TIME_hi'] = 0
condition = chefdf.loc[0:,'change_AVG_PREP_VID_TIME_hi'][chefdf['AVG_PREP_VID_TIME'] > change_AVG_PREP_VID_TIME_hi]

chefdf['change_AVG_PREP_VID_TIME_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# change_LARGEST_ORDER_SIZE_hi

chefdf['change_LARGEST_ORDER_SIZE_hi'] = 0
condition = chefdf.loc[0:,'change_LARGEST_ORDER_SIZE_hi'][chefdf['LARGEST_ORDER_SIZE'] > change_LARGEST_ORDER_SIZE_hi]

chefdf['change_LARGEST_ORDER_SIZE_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# change_AVG_CLICKS_PER_VISIT_hi

chefdf['change_AVG_CLICKS_PER_VISIT_hi'] = 0
condition = chefdf.loc[0:,'change_AVG_CLICKS_PER_VISIT_hi'][chefdf['AVG_CLICKS_PER_VISIT'] > change_AVG_CLICKS_PER_VISIT_hi]

chefdf['change_AVG_CLICKS_PER_VISIT_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

########################################
## change at threshold                ##
########################################

# change_MOBILE_NUMBER_at

chefdf['change_MOBILE_NUMBER_at'] = 0
condition = chefdf.loc[0:,'change_MOBILE_NUMBER_at'][chefdf['MOBILE_NUMBER'] == change_MOBILE_NUMBER_at ]

chefdf['change_MOBILE_NUMBER_at'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# change_TOTAL_PHOTOS_VIEWED_at

chefdf['change_TOTAL_PHOTOS_VIEWED_at'] = 0
condition = chefdf.loc[0:,'change_TOTAL_PHOTOS_VIEWED_at'][chefdf['TOTAL_PHOTOS_VIEWED'] == change_TOTAL_PHOTOS_VIEWED_at ]

chefdf['change_TOTAL_PHOTOS_VIEWED_at'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)


# change_WEEKLY_PLAN_change_at

chefdf['change_WEEKLY_PLAN_at'] = 0
condition = chefdf.loc[0:,'change_WEEKLY_PLAN_at'][chefdf['WEEKLY_PLAN'] == change_WEEKLY_PLAN_at ]

chefdf['change_WEEKLY_PLAN_at'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# change_TOTAL_PHOTOS_VIEWED_at

chefdf['change_TOTAL_PHOTOS_VIEWED_at'] = 0
condition = chefdf.loc[0:,'change_TOTAL_PHOTOS_VIEWED_at'][chefdf['TOTAL_PHOTOS_VIEWED'] == change_TOTAL_PHOTOS_VIEWED_at ]

chefdf['change_TOTAL_PHOTOS_VIEWED_at'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)
# change_UNIQUE_MEALS_PURCH_at

chefdf['change_UNIQUE_MEALS_PURCH_at'] = 0
condition = chefdf.loc[0:,'change_UNIQUE_MEALS_PURCH_at'][chefdf['UNIQUE_MEALS_PURCH'] == change_UNIQUE_MEALS_PURCH_at ]

chefdf['change_UNIQUE_MEALS_PURCH_at'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# change_MEDIAN_MEAL_RATING_at

chefdf['change_MEDIAN_MEAL_RATING_at'] = 0
condition = chefdf.loc[0:,'change_MEDIAN_MEAL_RATING_at'][chefdf['MEDIAN_MEAL_RATING'] == change_MEDIAN_MEAL_RATING_at ]

chefdf['change_MEDIAN_MEAL_RATING_at'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)
# change_CANCELLATIONS_AFTER_NOON_at
chefdf['change_CANCELLATIONS_AFTER_NOON_at'] = 0
condition = chefdf.loc[0:,'change_CANCELLATIONS_AFTER_NOON_at'][chefdf['CANCELLATIONS_AFTER_NOON'] == change_CANCELLATIONS_AFTER_NOON_at ]

chefdf['change_CANCELLATIONS_AFTER_NOON_at'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)


# Creating a placeholder list
placeholder_lst = []

# Loop over each email address
for index, col in chefdf.iterrows():
    
    # The email is splitted at '@'
    sp_email = chefdf.loc[index, 'EMAIL'].split(sep = "@")
    
    # The results are added to placeholder_lst
    placeholder_lst.append(sp_email)
    
# The placeholder_lst is converted into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)

# Renaming column
email_df.columns = ["name" , "personal_email"]

# Concatenating personal_email with friends DataFrame
chefdf = pd.concat([chefdf, email_df.loc[: , "personal_email"]],
                   axis = 1)

# email domain types

junk_email  = ['@me.com',
    '@aol.com', '@hotmail.com', '@live.com', '@msn.com', '@passport.com']

professional_email = ['@mmm.com',
    '@boeing.com', '@caterpillar.com', '@chevron.com', '@cisco.com', '@cocacola.com', 
    '@disney.com', '@dupont.com', '@exxon.com', '@ge.org', '@goldmansacs.com', 
    '@homedepot.com', '@ibm.com', '@intel.com', '@jnj.com', '@jpmorgan.com', 
    '@mcdonalds.com', '@merck.com', '@microsoft.com', '@nike.com', 
    '@pfizer.com', '@pg.com', '@travelers.com', '@unitedtech.com', '@unitedhealth.com', 
    '@verizon.com', '@visa.com', '@walmart.com', '@apple.com', '@amex.com']

personal_email = ['@gmail.com', '@yahoo.com', '@protonmail.com']

# placeholder list
placeholder_lst = []


# looping to group observations by domain type
for domain in chefdf['personal_email']:
        if '@'+ domain in personal_email:
            placeholder_lst.append('personal')
            
        elif '@'+ domain in professional_email:
            placeholder_lst.append('professional')
            
        elif '@'+ domain in junk_email:
            placeholder_lst.append('junk')
            
        else:
            print("Something went wrong, unknown value in mail")


# concatenating with original DataFrame
chefdf['domain_group'] = pd.Series(placeholder_lst)

#One hot encoding
one_hot_mail  = pd.get_dummies(chefdf['domain_group'])

# joining codings together
chefdf = chefdf.join([one_hot_mail])

# dropping variables 
chefdf = chefdf.drop(['EMAIL','domain_group', 'personal_email', 
                   'junk'], axis = 1)


#Scaling the data

scaling_data = chefdf

# Instantiating StandardScaler()
scaler = StandardScaler()

# Fitting the scaler with the scaled data
scaler.fit(scaling_data)

# Transform our data after fitting it
scaled_data = scaler.transform(scaling_data)

# Converting scaled data into a DataFrame
scaled_data_df = pd.DataFrame(scaled_data)

# Adding labels to the scaled DataFrame
scaled_data_df.columns = scaling_data.columns

# Checking the results
scaled_data_df.describe().round(2)

# Declaring response variable
chefdf_target = chefdf.loc[ : , 'CROSS_SELL_SUCCESS']

# Declaring explanatory variables
chefdf_data = scaled_data_df.drop('CROSS_SELL_SUCCESS', axis = 1)

# Perform a stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
            chefdf_data,
            chefdf_target,
            test_size = 0.25,
            random_state = 222,
            stratify = chefdf_target)

# Merge training data for statsmodels
chefdf_train = pd.concat([X_train, y_train], axis = 1)


# Declaring significant features in a list
significant_dict = {
'log_significant' : ['CANCELLATIONS_BEFORE_NOON',
                   'TASTES_AND_PREFERENCES','MOBILE_NUMBER','MOBILE_LOGINS',
                   'number_of_names', 'FOLLOWED_RECOMMENDATIONS_PCT','personal',
                   'professional']
}









# train/test split with the full model
chefdf_data   =  chefdf.loc[ : , significant_dict["log_significant"]]
chefdf_target =  chefdf.loc[ : , "CROSS_SELL_SUCCESS"]


# This is the exact code we were using before
X_train, X_test, y_train, y_test = train_test_split(
            chefdf_data,
            chefdf_target,
            test_size    = 0.25,
            random_state = 222,
            stratify     = chefdf_target)












# Instantiating a GradientBoostingClassifier model
gbm_default = GradientBoostingClassifier(random_state  = 222)

# Fitting training data
gbm_default_fit = gbm_default.fit(X_train, y_train)


# Predicitng on test set
gbm_default_pred = gbm_default_fit.predict(X_test)

############################"
#FINAL MODEL SCORE
###########################"
# Scoring and printing model results
print('Training ACCURACY:', gbm_default_fit.score(X_train, y_train).round(3))
print('Testing ACCURACY :', gbm_default_fit.score(X_test, y_test).round(3))
print('AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = gbm_default_pred).round(3))

test_score= roc_auc_score(y_true  = y_test,
                                          y_score = gbm_default_pred).round(3)


# In[ ]:




