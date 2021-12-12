import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import scipy.stats as stats


def stdP(x):
    """Calulates standart deviation

    Args:
        -x (pandas DataFrame): the first param is a DataFrame with all data

    Returns:
        -standart deviation
    """
    return np.std(x, ddof=0)

def seP(x):
    """Calulates standart error

    Args:
        -x (pandas DataFrame): the first param is a DataFrame with all data

    Returns:
        -standart error
    """
    return stats.sem(x, ddof=0)

def calcConvRate(df,column_group, column_target):
    """Take dataFrame and Calculate some statistic

    Args:
        - df (pandas DataFrame): the first param is a DataFrame with all data
        - column_group (string): the second param is a column which devides rows on main and control groups
        - column_target (string): third param is points to the target action need to be test

    Print:
        - pandas DataFrame where columns are:
            - total actions
            - sum of target actions
            - conversion rate
            - standart error

    """
    conversions = df.groupby(column_group)[column_target]
    conversion_rates = conversions.agg([np.mean, stdP, seP, 'count','sum'])

    results = conversion_rates.rename(columns={
    'count':'total_cnt'
    ,'sum':'target_cnt'
    ,'mean':'%conv_rate'
    ,'std_p':'deviation'
    ,'se_p':'error'}).style.format('{:.5f}')

    return results


def checkByZtest(df,column_group, column_target):
    """Test if two binomial distributions are statistically different from each other


    Args:
        - df (pandas DataFrame): the first param is a DataFrame with all data
        - column_group (string): the second param is a column which devides rows on main and control groups
        - column_target (string): third param is points to the target action need to be test

    Print:
        - z-score
        - p-value
        - confidence interval
    """

    results_cg = df[df[column_group] == 'control'][column_target]
    results_main = df[df[column_group] == 'main'][column_target]

    n_cg = results_cg.count()
    n_main = results_main.count()
    successes = [results_cg.sum(), results_main.sum()]
    nobs = [n_cg, n_main]

    z_stat, pval = proportions_ztest(successes, nobs=nobs)
    (lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs=nobs, alpha=0.05)

    print(f'z statistic: {z_stat:.4f}')
    print(f'p-value: {pval:.4f}')
    print(f'conf-interval 95% for control group: [{lower_con:.4f}, {upper_con:.4f}]')
    print(f'conf-interval 95% for main group: [{lower_treat:.4f}, {upper_treat:.4f}]')


def runLinePlotByCats(df, groupOfFileds, targetField):
    """Seaborn lineplot with multiple hue

    Args:
        - df (pandas DataFrame): the first param is a DataFrame with all data
        - groupOfFileds (array of string): the second param is an array of string values. Value is a
        field that make a category
        - column_target (string): third param is points to the target action need to be drawn

    Print:seaborn lineplot with multiple hue
    """

    fig, ax = plt.subplots(figsize=(15, 5))
    df['_'.join(groupOfFileds)] =  pd.Series(df.reindex(groupOfFileds,axis='columns').astype('str').values.tolist()).str.join('_')

   # PLOT WITH hue
    sns.lineplot(x='date', y=targetField, hue='_'.join(groupOfFileds), data=df,ax=ax)

    plt.title(f'Dynamic of {targetField} conversion rate')
    plt.show()

#     plt.clf()
#     plt.close()
