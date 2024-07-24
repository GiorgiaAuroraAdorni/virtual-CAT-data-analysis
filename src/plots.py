"""
@file-name:     plots.py
@date-creation: 09.07.2023
@author-name:   Giorgia Adorni
"""
import itertools
import os

import matplotlib.colors as colors
import matplotlib.image as mpimg
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import seaborn as sns
from matplotlib.patches import Patch
from rpy2.robjects import pandas2ri
from sklearn.metrics import confusion_matrix

from utils import import_png_images

pandas2ri.activate()

print(ro.r['version'])

plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
plt.rc('text', usetex=True)

sns.set('paper', 'white',
        rc={'legend.fontsize': 13, 'xtick.labelsize': 15, 'ytick.labelsize': 15,
            'axes.labelsize': 15, 'legend.title_fontsize': 14})
plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
plt.rc('text', usetex=True)


def plot_score_by_schema_and_group(df, y, y_label, category, output_directories):
    """
    Plot the score
    :param df:                 the DataFrame containing the data
    :param y:                  the variable to use for the y axis
    :param y_label:            the label of the y axis
    :param category:           the category to use for grouping the data
    :param output_directories: the directories where to save the plots
    """
    df = df.copy()
    df = df[['STUDENT_ID', 'SCHEMA_ID', y, category]]
    print(df[y].describe())

    if category == 'AGE_CATEGORY':
        df[category] = pd.Categorical(df[category], categories=['From 3 to 6 years old',
                                                                'From 7 to 9 years old',
                                                                'From 10 to 13 years old',
                                                                'From 14 to 16 years old'])
        df[category] = df[category].cat.rename_categories(['3-6 yrs', '7-9 yrs', '10-13 yrs', '14-16 yrs'])

    plt.figure(figsize=(6, 4))

    # Create a plot with the task on the x-axis and the score on the y-axis, grouped by category
    ax = sns.lineplot(x='SCHEMA_ID', y=y, hue=category, data=df, estimator='mean',
                      marker='o', markersize=5, dashes=False)

    plt.xlabel('Schema')
    plt.ylabel(y_label)

    # Show all the x-ticks
    ax.set_xticks(range(1, 13))

    if y in ['CAT_SCORE', 'WEIGHTED_CAT_SCORE']:
        # Set te y ticks from 0 to 5
        ax.set_yticks(range(0, 6))
    else:
        # Set te y ticks from 0 to 3
        ax.set_yticks(range(0, 3))

    leg = ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', borderaxespad=0.,
                    handletextpad=0.5, handlelength=1)
    leg.set_title('Age\ncategory')
    leg.get_title().set_multialignment('center')

    for out_dir in output_directories:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        extension = out_dir.split('/')[-2]

        plt.savefig(out_dir + 'line_{}_by_task_and_{}.{}'.format(y, category, extension), dpi=600, transparent=True,
                    bbox_inches='tight', metadata={'Author': 'Giorgia Adorni'})

    plt.close()


def plot_score_by_group(df, y, y_label, category, output_directories):
    """
    Plot the score by group
    :param df:                 the DataFrame containing the data
    :param y:                  the variable to use for the y axis
    :param y_label:            the label of the y axis
    :param category:           the category to use for grouping the data
    :param output_directories: the directories where to save the plots
    """
    df = df.copy()
    df = df[['STUDENT_ID', y, category]]

    print(df[y].describe())

    if category == 'AGE_CATEGORY':
        df[category] = pd.Categorical(df[category], categories=['From 3 to 6 years old',
                                                                'From 7 to 9 years old',
                                                                'From 10 to 13 years old',
                                                                'From 14 to 16 years old'])
        df[category] = df[category].cat.rename_categories(['3-6 yrs', '7-9 yrs', '10-13 yrs', '14-16 yrs'])

    plt.figure(figsize=(6, 4))

    # Calculate the average scores for each student
    df['Average_Score'] = df.groupby('STUDENT_ID')[y].transform('mean')

    # Sort the DataFrame by category and average score
    df.sort_values(by=[category, 'Average_Score'], inplace=True)

    # Assign a new ID based on average score within each category, starting from 0
    df['Category_New_ID'] = df.groupby(category, observed=False)['Average_Score'].rank(method='dense', ascending=True)

    # TODO
    # Find the maximum new ID for each category to normalize the scale
    max_ids_per_category = df.groupby(category, observed=False)['Category_New_ID'].transform('max')

    # Normalize each student's new ID within the category
    df['Normalized_Category_ID'] = df['Category_New_ID'] / max_ids_per_category

    # Optional: rescale the normalized IDs for better visualization
    # For example, if you want to scale it up to the number of students in the largest category
    largest_category_size = max_ids_per_category.max()
    df['Scaled_Category_ID'] = df['Normalized_Category_ID'] * largest_category_size

    ax = sns.lineplot(x='Scaled_Category_ID', y=y, hue=category, data=df,
                      marker='o', markersize=5, dashes=False, estimator='mean')

    plt.xlabel('Student')
    plt.ylabel(y_label)

    if y in ['CAT_SCORE', 'WEIGHTED_CAT_SCORE']:
        # Set te y ticks from 0 to 5
        ax.set_yticks(range(0, 6))
    else:
        # Set te y ticks from 0 to 3
        ax.set_yticks(range(0, 3))

    # Set the x-ticks from 0 to the number of students in the largest category, and show every 5 ticks
    # I want to substitute largest_category_size with a multiple of 5 that is greater than largest_category_size
    new_largest_category_size = largest_category_size + (5 - largest_category_size % 5)
    ax.set_xticks(range(0, int(new_largest_category_size), 5))

    leg = ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', borderaxespad=0.,
                    handletextpad=0.5, handlelength=1)
    leg.set_title('Age\ncategory')
    leg.get_title().set_multialignment('center')

    for out_dir in output_directories:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        extension = out_dir.split('/')[-2]

        plt.savefig(out_dir + 'line_{}_by_{}.{}'.format(y,category, extension), dpi=600, transparent=True,
                    bbox_inches='tight', metadata={'Author': 'Giorgia Adorni'})

    plt.close()


def plot_interaction_dimension(df, output_directories):
    """
    Plot the interaction dimension
    :param df:                 the DataFrame containing the data
    :param output_directories: the directories where to save the plots
    :return logs_df:           the DataFrame containing the data without nans
    """
    # Create a new column PREDOMINANT_ARTEFACT_TYPE using a mapping from PREDOMINANT_INTERFACE and PREDOMINANT_FEEDBACK
    # If PREDOMINANT_INTERFACE is 0 and PREDOMINANT_FEEDBACK is True 0 (GF),
    # If PREDOMINANT_INTERFACE is 0 and PREDOMINANT_FEEDBACK is False 1 (G),
    # If PREDOMINANT_INTERFACE is 1 and PREDOMINANT_FEEDBACK is True 2 (PF),
    # If PREDOMINANT_INTERFACE is 1 and PREDOMINANT_FEEDBACK is False 3 (P)
    df['PREDOMINANT_ARTEFACT_TYPE'] = df.apply(
        lambda x: 'GF' if x['PREDOMINANT_INTERFACE'] == 0 and x['PREDOMINANT_FEEDBACK'] is True else
        'G' if x['PREDOMINANT_INTERFACE'] == 0 and x['PREDOMINANT_FEEDBACK'] is False else
        'PF' if x['PREDOMINANT_INTERFACE'] == 1 and x['PREDOMINANT_FEEDBACK'] is True else
        'P' if x['PREDOMINANT_INTERFACE'] == 1 and x['PREDOMINANT_FEEDBACK'] is False else
        np.nan, axis=1)

    # Rename RETRIEVED_ARTEFACT_TYPE with a mapping
    df['RETRIEVED_ARTEFACT_TYPE'] = df['RETRIEVED_ARTEFACT_TYPE'].map({
        '0 (GF)': 'GF',
        '1 (G)': 'G',
        '2 (PF)': 'PF',
        '3 (P)': 'P'
    })

    # Add the PREDOMINANT_ARTEFACT_DIMENSION column to the DataFrame
    df['PREDOMINANT_ARTEFACT_DIMENSION'] = df['PREDOMINANT_ARTEFACT_TYPE'].map({
        'GF': 0,
        'G': 1,
        'PF': 2,
        'P': 3
    })

    # Compare the RETRIEVED_ARTEFACT_TYPE and PREDOMINANT_ARTEFACT_TYPE columns
    # and print the number of rows where RETRIEVED_ARTEFACT_TYPE is different from PREDOMINANT_ARTEFACT_TYPE
    print(df[df['RETRIEVED_ARTEFACT_TYPE'] != df['PREDOMINANT_ARTEFACT_TYPE']].shape[0])

    # Remove nans
    logs_df = df.dropna(subset=['RETRIEVED_ARTEFACT_TYPE', 'PREDOMINANT_ARTEFACT_TYPE'])

    artefact_order = ['GF', 'G', 'PF', 'P']

    conf_mat = confusion_matrix(logs_df['RETRIEVED_ARTEFACT_TYPE'], logs_df['PREDOMINANT_ARTEFACT_TYPE'],
                                labels=artefact_order)

    # Convert the confusion matrix to percentages
    conf_mat_percentage = conf_mat / np.sum(conf_mat) * 100
    print(conf_mat_percentage)

    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(conf_mat_percentage, annot=True, fmt='', cmap='Blues',
                     xticklabels=artefact_order, yticklabels=artefact_order, annot_kws={'size': 14}, square=True,
                     cbar_kws={'shrink': 0.74, 'pad': 0.125}, vmin=0, vmax=30)

    for t in ax.texts:
        t.set_text("{:.0f}\%".format(float(t.get_text())))

    # Get the colorbar object from the Seaborn heatmap
    cbar = ax.collections[0].colorbar

    # Set the colorbar ticks and labels as percentage
    cbar_ticks = np.arange(0, 31, 10)  # Ticks every 10%
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{tick}\%' for tick in cbar_ticks])

    # Invert y-axis
    plt.gca().invert_yaxis()

    # Compute the actual total percentage
    row_percentages = conf_mat_percentage.sum(axis=1)
    annot_rows = [f"{num}\%" for num in row_percentages.round().astype(int).astype(str)]

    # Add it percentage to the plot
    nrows = conf_mat_percentage.shape[0]
    for i in range(nrows):
        # Compute the y position of the text and add 0.5 to center it in the row
        y_pos = i + 0.5
        ax.text(ax.get_xlim()[1] + 0.3, y_pos, annot_rows[i], size=14, ha='center', va='center')

    # Compute the prevalent total percentage
    columns_percentages = conf_mat_percentage.sum(axis=0)
    annot_columns = [f"{num}\%" for num in columns_percentages.round().astype(int).astype(str)]

    # Add it to the plot
    ncols = conf_mat_percentage.shape[1]
    for j in range(ncols):
        # Compute the x position of the text and add 0.5 to center it in the column
        x_pos = j + 0.5
        ax.text(x_pos, ax.get_ylim()[1] + 0.2, annot_columns[j], size=14, ha='center', va='center')

    plt.xlabel('Prevalent interaction dimension')
    plt.ylabel('Lowest interaction dimension')

    for out_dir in output_directories:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        extension = out_dir.split('/')[-2]

        plt.savefig(out_dir + 'used_interaction_dimensions.{}'.format(extension), dpi=600, transparent=True,
                    bbox_inches='tight', metadata={'Author': 'Giorgia Adorni'})

    plt.close()

    return logs_df


def generate_time_plot(df, output_directories):
    # Plot setup
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)  # 2 rows, 1 column, with a specified figure size

    # # Make the age category and artefact category categorical
    df['AGE_CATEGORY'] = pd.Categorical(df['AGE_CATEGORY'], categories=['From 3 to 6 years old',
                                                                        'From 7 to 9 years old',
                                                                        'From 10 to 13 years old',
                                                                        'From 14 to 16 years old'])
    # Rename age categories in 3-6 yrs, 7-9 yrs, 10-13 yrs, 14-16 yrs
    df['AGE_CATEGORY'] = df['AGE_CATEGORY'].cat.rename_categories(['3-6 yrs', '7-9 yrs', '10-13 yrs', '14-16 yrs'])
    df['ARTEFACT_TYPE'] = pd.Categorical(df['ARTEFACT_TYPE'], categories=['GF', 'G', 'PF', 'P'])

    # df['SCHEMA_ID'] = pd.Categorical(df['SCHEMA_ID'], categories=['1', '2', '3', '4', '5', '6',
    #                                                               '7', '8', '9', '10', '11', '12'])

    # Transform the time in minutes
    df['LOG_TIME'] = df['TIME'] / 60

    # Create subplot for age category
    sns.stripplot(x='SCHEMA_ID', y='LOG_TIME', hue='AGE_CATEGORY', data=df,
                  ax=axs[0], dodge=True, alpha=0.5, size=5)
    sns.boxplot(x='SCHEMA_ID', y='LOG_TIME', hue='AGE_CATEGORY', data=df,
                ax=axs[0], dodge=True, legend=False, boxprops=dict(alpha=0.5), showfliers=False, whis=0)

    # Add the median as a square in the first subplot
    medians_age = df.groupby(['SCHEMA_ID', 'AGE_CATEGORY'], observed=True)['LOG_TIME'].median().reset_index()
    sns.stripplot(x='SCHEMA_ID', y='LOG_TIME', hue='AGE_CATEGORY', data=medians_age,
                  ax=axs[0], dodge=True, alpha=1, marker='s', legend=False, size=6)

    # Add a line to plot the average time
    # average_age = df.groupby(['SCHEMA_ID', 'AGE_CATEGORY'], observed=True)['LOG_TIME'].mean().reset_index()
    # sns.lineplot(x='SCHEMA_ID', y='LOG_TIME', hue='AGE_CATEGORY', data=average_age,
    #              ax=axs[0], alpha=1, legend=False)

    # Create subplot for artefact category
    sns.stripplot(x='SCHEMA_ID', y='LOG_TIME', hue='ARTEFACT_TYPE', data=df,
                  ax=axs[1], dodge=True, alpha=0.5, size=5)
    sns.boxplot(x='SCHEMA_ID', y='LOG_TIME', hue='ARTEFACT_TYPE', data=df,
                ax=axs[1], dodge=True, legend=False, boxprops=dict(alpha=0.5), showfliers=False, whis=0)

    # Add the median as a square in the second subplot
    medians_artefact = df.groupby(['SCHEMA_ID', 'ARTEFACT_TYPE'], observed=True)['LOG_TIME'].median().reset_index()
    sns.stripplot(x='SCHEMA_ID', y='LOG_TIME', hue='ARTEFACT_TYPE', data=medians_artefact,
                  ax=axs[1], dodge=True, alpha=1, marker='s', legend=False, size=6)

    # Add a line to plot the average time
    # average_artefact = df.groupby(['SCHEMA_ID', 'ARTEFACT_TYPE'], observed=True)['LOG_TIME'].mean().reset_index()
    # sns.lineplot(x='SCHEMA_ID', y='LOG_TIME', hue='ARTEFACT_TYPE', data=average_artefact,
    #              ax=axs[1], alpha=1, legend=False, estimator='mean')

    # Adjust layout
    plt.xlabel('Schema')
    axs[0].set_ylabel('Time (minutes)')
    axs[1].set_ylabel('Time (minutes)')

    # Move both legends to the right, outside the plot area, centered vertically
    # axs[0].legend(bbox_to_anchor=(1.01, 0.5), loc='center left', borderaxespad=0., title='Age\nCategory')
    # axs[1].legend(bbox_to_anchor=(1.01, 0.5), loc='center left', borderaxespad=0., title='Artefact\nCategory')

    leg = axs[0].legend(bbox_to_anchor=(1.01, 0.5), loc='center left', borderaxespad=0.,
                        handletextpad=0.5, handlelength=1)
    leg.set_title('Age\ncategory')
    leg.get_title().set_multialignment('center')

    leg = axs[1].legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0.,
                        handletextpad=0.5, handlelength=1)
    leg.set_title('Interaction\ndimension',)
    leg.get_title().set_multialignment('center')

    plt.tight_layout()

    print(medians_age)
    print(medians_artefact)

    for out_dir in output_directories:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        extension = out_dir.split('/')[-2]

        plt.savefig(out_dir + 'time_by_categories.{}'.format(extension), dpi=600, transparent=True,
                    bbox_inches='tight', metadata={'Author': 'Giorgia Adorni'})

    plt.close()


def generate_restart_distribution_plot(df, column, x_label, output_directories, order=None, x_ticks=None):
    """
    Generate a distribution plot
    :param df:                 the DataFrame containing the data
    :param column:             the column to use
    :param x_label:            the x label
    :param output_directories: the directories where to save the plots
    :param order:              the order of the x-axis
    :param x_ticks:            the labels of the x-axis
    """
    # Calculate the mean and count for each category in 'column'
    grouped = df.groupby(column)['RESTARTS'].agg(['mean', 'count'])

    # Reindex based on the 'order' list
    grouped = grouped.reindex(order)

    # Calculate the total count across all categories
    total_count = grouped['count'].sum()

    # Number of categories
    n_categories = len(order)

    # Calculate the new metric (scaled mean)
    grouped['scaled_mean'] = grouped['mean'] * (grouped['count'] / total_count) * n_categories

    # Create a bar plot
    plt.figure(figsize=(6, 5))

    g = sns.barplot(x=grouped.index, y=grouped['scaled_mean'], color='lightblue')

    # Add a line plot for the scaled mean
    plt.plot(range(len(order)), grouped['scaled_mean'].values, color='red', marker='o')

    # Add text labels for each point
    for i, value in enumerate(grouped['scaled_mean'].values):
        value_text = round(value, 2)
        plt.text(i, value + 0.01, value_text, ha='center', size=13)

    plt.xlabel(x_label)
    plt.ylabel('Avg number of restarts')

    plt.ylim(0, 0.53)

    if x_ticks is not None:
        g.set_xticks(range(len(order)))
        g.set_xticklabels(x_ticks)

    for out_dir in output_directories:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        extension = out_dir.split('/')[-2]

        plt.savefig(out_dir + 'restart_distribution_{}.{}'.format(x_label, extension), dpi=600, transparent=True,
                    bbox_inches='tight', metadata={'Author': 'Giorgia Adorni'})

    plt.close()


def plot_group_frequencies(survey_data_merged, survey_questions, survey_questions_replacement, answers_order,
                           output_folders):
    """
    Plot the frequency of responses for each question
    :param survey_data_merged:           the DataFrame containing the survey data
    :param survey_questions:             the questions of the survey
    :param survey_questions_replacement: the replacement for the questions of the survey
    :param answers_order:                the order of the answers for each question
    :param output_folders:               the directories where to save the plots
    """
    responses_frequency_hue(survey_data_merged, survey_questions, survey_questions_replacement, answers_order,
                            'Session', 'SESSION_ID', output_folders)

    responses_frequency_hue(survey_data_merged, survey_questions, survey_questions_replacement, answers_order,
                            'Age', 'AGE_GROUP', output_folders)

    responses_frequency_hue(survey_data_merged, survey_questions, survey_questions_replacement, answers_order,
                            'Gender', 'GENDER', output_folders)

    responses_frequency_hue(survey_data_merged, survey_questions, survey_questions_replacement, answers_order,
                            'Canton', 'CANTON_NAME', output_folders)

    responses_frequency_hue(survey_data_merged, survey_questions, survey_questions_replacement, answers_order,
                            'Average CAT score', 'AVG_CAT_SCORE', output_folders)

    responses_frequency_hue(survey_data_merged, survey_questions, survey_questions_replacement, answers_order,
                            'Average algorithm dimension', 'AVG_ALGORITHM_DIMENSION', output_folders)


def plot_crosstabs(survey_data_merged, survey_questions_replacement, answers_order, score_labels, algorithm_labels, output_folders):
    """
    Plot the cross-tabulation heatmaps
    :param survey_data_merged:           the DataFrame containing the survey data
    :param survey_questions_replacement: the replacement for the questions of the survey
    :param answers_order:                the order of the answers for each question
    :param score_labels:                 the labels for the score
    :param algorithm_labels:             the labels for the algorithm dimension
    :param output_folders:               the directories where to save the plots
    """
    # multiple_crosstab_heatmap(xs=[survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
    #                               survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
    #                               survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
    #                               survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES']],
    #                           y=survey_data_merged['DID_YOU_ENJOY_THIS_ACTIVITY'],
    #                           x_labels=[survey_questions_replacement[2],
    #                                     survey_questions_replacement[3],
    #                                     survey_questions_replacement[5],
    #                                     survey_questions_replacement[6]],
    #                           y_label=survey_questions_replacement[0],
    #                           x_orders=[answers_order[2],
    #                                     answers_order[3],
    #                                     answers_order[5],
    #                                     answers_order[6]],
    #                           y_order=answers_order[0],
    #                           output_directories=output_folders,
    #                           number=1)
    #
    # multiple_crosstab_heatmap(xs=[survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
    #                               survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
    #                               survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
    #                               survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES']],
    #                           y=survey_data_merged['HAVE_YOU_EVER_USED_AN_APP_LIKE_THIS_TO_DO_EXERCISES_AND_LEARN'],
    #                           x_labels=[survey_questions_replacement[2],
    #                                     survey_questions_replacement[3],
    #                                     survey_questions_replacement[5],
    #                                     survey_questions_replacement[6]],
    #                           y_label=survey_questions_replacement[1],
    #                           x_orders=[answers_order[2],
    #                                     answers_order[3],
    #                                     answers_order[5],
    #                                     answers_order[6]],
    #                           y_order=answers_order[1],
    #                           output_directories=output_folders,
    #                           number=2)
    #
    # multiple_crosstab_heatmap(xs=[survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
    #                               survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
    #                               survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
    #                               survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES']],
    #                           y=survey_data_merged['WHICH_RESOLUTION_MODE_DID_YOU_PREFER_TO_USE'],
    #                           x_labels=[survey_questions_replacement[2],
    #                                     survey_questions_replacement[3],
    #                                     survey_questions_replacement[5],
    #                                     survey_questions_replacement[6]],
    #                           y_label=survey_questions_replacement[4],
    #                           x_orders=[answers_order[2],
    #                                     answers_order[3],
    #                                     answers_order[5],
    #                                     answers_order[6]],
    #                           y_order=answers_order[4],
    #                           output_directories=output_folders,
    #                           number=3)
    #
    # multiple_crosstab_heatmap(xs=[survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
    #                               survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
    #                               survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
    #                               survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES']],
    #                           y=survey_data_merged['WOULD_YOU_DO_THIS_EXPERIENCE_AGAIN'],
    #                           x_labels=[survey_questions_replacement[2],
    #                                     survey_questions_replacement[3],
    #                                     survey_questions_replacement[5],
    #                                     survey_questions_replacement[6]],
    #                           y_label=survey_questions_replacement[7],
    #                           x_orders=[answers_order[2],
    #                                     answers_order[3],
    #                                     answers_order[5],
    #                                     answers_order[6]],
    #                           y_order=answers_order[7],
    #                           output_directories=output_folders,
    #                           number=4)
    #
    multiple_crosstab_heatmap(xs=[survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
                                  survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
                                  survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
                                  survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES']],
                              y=survey_data_merged['AVG_CAT_SCORE'],
                              x_labels=[survey_questions_replacement[2],
                                        survey_questions_replacement[3],
                                        survey_questions_replacement[5],
                                        survey_questions_replacement[6]],
                              y_label='Average CAT score',
                              x_orders=[answers_order[2],
                                        answers_order[3],
                                        answers_order[5],
                                        answers_order[6]],
                              y_order=score_labels,
                              output_directories=output_folders,
                              number=5)

    multiple_crosstab_heatmap(xs=[survey_data_merged['DID_YOU_ENJOY_THIS_ACTIVITY'],
                                  survey_data_merged['HAVE_YOU_EVER_USED_AN_APP_LIKE_THIS_TO_DO_EXERCISES_AND_LEARN'],
                                  survey_data_merged['WHICH_RESOLUTION_MODE_DID_YOU_PREFER_TO_USE'],
                                  survey_data_merged['WOULD_YOU_DO_THIS_EXPERIENCE_AGAIN']],
                              y=survey_data_merged['AVG_CAT_SCORE'],
                              x_labels=[survey_questions_replacement[0],
                                        survey_questions_replacement[1],
                                        survey_questions_replacement[4],
                                        survey_questions_replacement[7]],
                              y_label='Average CAT score',
                              x_orders=[answers_order[0],
                                        answers_order[1],
                                        answers_order[4],
                                        answers_order[7]],
                              y_order=score_labels,
                              output_directories=output_folders,
                              number=6)

    multiple_crosstab_heatmap(xs=[survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
                                  survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
                                  survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
                                  survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES']],
                              y=survey_data_merged['AVG_ALGORITHM_DIMENSION'],
                              x_labels=[survey_questions_replacement[2],
                                        survey_questions_replacement[3],
                                        survey_questions_replacement[5],
                                        survey_questions_replacement[6]],
                              y_label='Average algorithm dimension',
                              x_orders=[answers_order[2],
                                        answers_order[3],
                                        answers_order[5],
                                        answers_order[6]],
                              y_order=algorithm_labels,
                              output_directories=output_folders,
                              number=7)

    multiple_crosstab_heatmap(xs=[survey_data_merged['DID_YOU_ENJOY_THIS_ACTIVITY'],
                                  survey_data_merged['HAVE_YOU_EVER_USED_AN_APP_LIKE_THIS_TO_DO_EXERCISES_AND_LEARN'],
                                  survey_data_merged['WHICH_RESOLUTION_MODE_DID_YOU_PREFER_TO_USE'],
                                  survey_data_merged['WOULD_YOU_DO_THIS_EXPERIENCE_AGAIN']],
                              y=survey_data_merged['AVG_ALGORITHM_DIMENSION'],
                              x_labels=[survey_questions_replacement[0],
                                        survey_questions_replacement[1],
                                        survey_questions_replacement[4],
                                        survey_questions_replacement[7]],
                              y_label='Average algorithm dimension',
                              x_orders=[answers_order[0],
                                        answers_order[1],
                                        answers_order[4],
                                        answers_order[7]],
                              y_order=algorithm_labels,
                              output_directories=output_folders,
                              number=8)


    # # Activity enjoyment
    # crosstab_heatmap(survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
    #                  survey_data_merged['DID_YOU_ENJOY_THIS_ACTIVITY'],
    #                  survey_questions_replacement[2], survey_questions_replacement[0],
    #                  answers_order[2], answers_order[0],
    #                  # 'Relationship between ease of use and enjoyment of the activity',
    #                  output_folders, 1)
    #
    # crosstab_heatmap(survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
    #                  survey_data_merged['DID_YOU_ENJOY_THIS_ACTIVITY'],
    #                  survey_questions_replacement[3], survey_questions_replacement[0],
    #                  answers_order[3], answers_order[0],
    #                  # 'Relationship between CAT score and enjoyment of the activity',
    #                  output_folders, 2)
    #
    # crosstab_heatmap(survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
    #                  survey_data_merged['DID_YOU_ENJOY_THIS_ACTIVITY'],
    #                  survey_questions_replacement[5], survey_questions_replacement[0],
    #                  answers_order[5], answers_order[0],
    #                  # 'Relationship between ease of exercise and enjoyment of the activity',
    #                  output_folders, 3)
    #
    # crosstab_heatmap(survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES'],
    #                  survey_data_merged['DID_YOU_ENJOY_THIS_ACTIVITY'],
    #                  survey_questions_replacement[6], survey_questions_replacement[0],
    #                  answers_order[6], answers_order[0],
    #                  # 'Relationship between how long it took to complete the exercises and enjoyment of the activity',
    #                  output_folders, 4)
    #
    # crosstab_heatmap(survey_data_merged['AVG_CAT_SCORE'],
    #                  survey_data_merged['DID_YOU_ENJOY_THIS_ACTIVITY'],
    #                  'Average CAT score', survey_questions_replacement[0],
    #                  score_labels, answers_order[0],
    #                  # 'Relationship between CAT score and enjoyment of the activity',
    #                  output_folders, 5)
    #
    # # Prior experience
    # crosstab_heatmap(survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
    #                  survey_data_merged['HAVE_YOU_EVER_USED_AN_APP_LIKE_THIS_TO_DO_EXERCISES_AND_LEARN'],
    #                  survey_questions_replacement[2], survey_questions_replacement[1],
    #                  answers_order[2], answers_order[1],
    #                  # 'Relationship between ease of use and previous experience',
    #                  output_folders, 6)
    #
    # # Relationship between ease of use and previous experience
    # crosstab_heatmap(survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
    #                  survey_data_merged['HAVE_YOU_EVER_USED_AN_APP_LIKE_THIS_TO_DO_EXERCISES_AND_LEARN'],
    #                  survey_questions_replacement[3], survey_questions_replacement[1],
    #                  answers_order[3], answers_order[1],
    #                  # 'Relationship between ease of use and previous experience',
    #                  output_folders, 7)
    #
    # crosstab_heatmap(survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
    #                  survey_data_merged['HAVE_YOU_EVER_USED_AN_APP_LIKE_THIS_TO_DO_EXERCISES_AND_LEARN'],
    #                  survey_questions_replacement[5], survey_questions_replacement[1],
    #                  answers_order[5], answers_order[1],
    #                  # 'Relationship between ease of exercise and previous experience',
    #                  output_folders, 8)
    #
    # crosstab_heatmap(survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES'],
    #                  survey_data_merged['HAVE_YOU_EVER_USED_AN_APP_LIKE_THIS_TO_DO_EXERCISES_AND_LEARN'],
    #                  survey_questions_replacement[6], survey_questions_replacement[1],
    #                  answers_order[6], answers_order[1],
    #                  # 'Relationship between how long it took to complete the exercises and previous experience',
    #                  output_folders, 9)
    #
    # crosstab_heatmap(survey_data_merged['AVG_CAT_SCORE'],
    #                  survey_data_merged['HAVE_YOU_EVER_USED_AN_APP_LIKE_THIS_TO_DO_EXERCISES_AND_LEARN'],
    #                  'Average CAT score', survey_questions_replacement[1],
    #                  score_labels, answers_order[1],
    #                  # 'Relationship between CAT score and previous experience',
    #                  output_folders, 10)
    #
    # # Interaction preference
    # crosstab_heatmap(survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
    #                  survey_data_merged['WHICH_RESOLUTION_MODE_DID_YOU_PREFER_TO_USE'],
    #                  survey_questions_replacement[2], survey_questions_replacement[4],
    #                  answers_order[2], answers_order[4],
    #                  # 'Relationship between ease of use and resolution mode',
    #                  output_folders, 11)
    #
    # crosstab_heatmap(survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
    #                  survey_data_merged['WHICH_RESOLUTION_MODE_DID_YOU_PREFER_TO_USE'],
    #                  survey_questions_replacement[3], survey_questions_replacement[4],
    #                  answers_order[3], answers_order[4],
    #                  # 'Relationship between ease of use and resolution mode',
    #                  output_folders, 12)
    #
    # crosstab_heatmap(survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
    #                  survey_data_merged['WHICH_RESOLUTION_MODE_DID_YOU_PREFER_TO_USE'],
    #                  survey_questions_replacement[5], survey_questions_replacement[4],
    #                  answers_order[5], answers_order[4],
    #                  # 'Relationship between ease of exercise and resolution mode',
    #                  output_folders, 13)
    #
    #
    # crosstab_heatmap(survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES'],
    #                  survey_data_merged['WHICH_RESOLUTION_MODE_DID_YOU_PREFER_TO_USE'],
    #                  survey_questions_replacement[6], survey_questions_replacement[4],
    #                  answers_order[6], answers_order[4],
    #                  # 'Relationship between how long it took to complete the exercises and resolution mode',
    #                  output_folders, 14)
    #
    # crosstab_heatmap(survey_data_merged['AVG_CAT_SCORE'],
    #                  survey_data_merged['WHICH_RESOLUTION_MODE_DID_YOU_PREFER_TO_USE'],
    #                  'Average CAT score', survey_questions_replacement[4],
    #                  score_labels, answers_order[4],
    #                  # 'Relationship between CAT score and resolution mode',
    #                  output_folders, 15)
    #
    # # Willingness to repeat the experience
    # crosstab_heatmap(survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
    #                  survey_data_merged['WOULD_YOU_DO_THIS_EXPERIENCE_AGAIN'],
    #                  survey_questions_replacement[2], survey_questions_replacement[7],
    #                  answers_order[2], answers_order[7],
    #                  # 'Relationship between ease of use and resolution mode',
    #                  output_folders, 16)
    #
    # crosstab_heatmap(survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
    #                  survey_data_merged['WOULD_YOU_DO_THIS_EXPERIENCE_AGAIN'],
    #                  survey_questions_replacement[3], survey_questions_replacement[7],
    #                  answers_order[3], answers_order[7],
    #                  # 'Relationship between ease of use and resolution mode',
    #                  output_folders, 17)
    #
    # crosstab_heatmap(survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
    #                  survey_data_merged['WOULD_YOU_DO_THIS_EXPERIENCE_AGAIN'],
    #                  survey_questions_replacement[5], survey_questions_replacement[7],
    #                  answers_order[5], answers_order[7],
    #                  # 'Relationship between ease of exercise and resolution mode',
    #                  output_folders, 18)
    #
    # crosstab_heatmap(survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES'],
    #                  survey_data_merged['WOULD_YOU_DO_THIS_EXPERIENCE_AGAIN'],
    #                  survey_questions_replacement[6], survey_questions_replacement[7],
    #                  answers_order[6], answers_order[7],
    #                  # 'Relationship between how long it took to complete the exercises and resolution mode',
    #                  output_folders, 19)
    #
    # crosstab_heatmap(survey_data_merged['AVG_CAT_SCORE'],
    #                  survey_data_merged['WOULD_YOU_DO_THIS_EXPERIENCE_AGAIN'],
    #                  'Average CAT score', survey_questions_replacement[7],
    #                  score_labels, answers_order[7],
    #                  # 'Relationship between CAT score and resolution mode',
    #                  output_folders, 20)

    # Self-efficacy
    # App ease of use
    crosstab_heatmap(survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
                     survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
                     survey_questions_replacement[3], survey_questions_replacement[2],
                     answers_order[3], answers_order[2],
                     # 'Relationship between how long it took to complete the exercises and ease of use',
                     output_folders, 21)

    crosstab_heatmap(survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
                     survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
                     survey_questions_replacement[5], survey_questions_replacement[2],
                     answers_order[5], answers_order[2],
                     # 'Relationship between how long it took to complete the exercises and ease of use',
                     output_folders, 22)

    crosstab_heatmap(survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES'],
                     survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
                     survey_questions_replacement[6], survey_questions_replacement[2],
                     answers_order[6], answers_order[2],
                     # 'Relationship between how long it took to complete the exercises and ease of use',
                     output_folders, 23)

    # Rules clarity
    crosstab_heatmap(survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
                     survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
                     survey_questions_replacement[2], survey_questions_replacement[3],
                     answers_order[2], answers_order[3],
                     # 'Relationship between how long it took to complete the exercises and ease of exercise',
                     output_folders, 24)

    crosstab_heatmap(survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
                     survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
                     survey_questions_replacement[5], survey_questions_replacement[3],
                     answers_order[5], answers_order[3],
                     # 'Relationship between how long it took to complete the exercises and ease of exercise',
                     output_folders, 25)

    crosstab_heatmap(survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES'],
                     survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
                     survey_questions_replacement[6], survey_questions_replacement[3],
                     answers_order[6], answers_order[3],
                     # 'Relationship between how long it took to complete the exercises and ease of exercise',
                     output_folders, 26)

    # Exercise difficulty
    crosstab_heatmap(survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
                     survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
                     survey_questions_replacement[2], survey_questions_replacement[5],
                     answers_order[2], answers_order[5],
                     # 'Relationship between how long it took to complete the exercises and ease of exercise',
                     output_folders, 27)

    crosstab_heatmap(survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
                     survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
                     survey_questions_replacement[3], survey_questions_replacement[5],
                     answers_order[3], answers_order[5],
                     # 'Relationship between how long it took to complete the exercises and ease of exercise',
                     output_folders, 28)

    crosstab_heatmap(survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES'],
                     survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
                     survey_questions_replacement[6], survey_questions_replacement[5],
                     answers_order[6], answers_order[5],
                     # 'Relationship between how long it took to complete the exercises and ease of exercise',
                     output_folders, 29)

    # Completion time
    crosstab_heatmap(survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
                     survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES'],
                     survey_questions_replacement[2], survey_questions_replacement[6],
                     answers_order[2], answers_order[6],
                     # 'Relationship between how long it took to complete the exercises and ease of exercise',
                     output_folders, 30)

    crosstab_heatmap(survey_data_merged['WERE_THE_RULES_OF_THE_ACTIVITY_EASY_TO_UNDERSTAND'],
                     survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES'],
                     survey_questions_replacement[3], survey_questions_replacement[6],
                     answers_order[3], answers_order[6],
                     # 'Relationship between how long it took to complete the exercises and ease of exercise',
                     output_folders, 31)

    crosstab_heatmap(survey_data_merged['WERE_THE_EXERCISES_EASY_TO_SOLVE'],
                     survey_data_merged['HOW_LONG_DID_YOU_TAKE_TO_COMPLETE_THE_EXERCISES'],
                     survey_questions_replacement[5], survey_questions_replacement[6],
                     answers_order[5], answers_order[6],
                     # 'Relationship between how long it took to complete the exercises and ease of exercise',
                     output_folders, 32)

    # # Relationship between cat score and ease of use
    # crosstab_heatmap(survey_data_merged['AVG_CAT_SCORE'],
    #                  survey_data_merged['WAS_THE_APP_EASY_TO_USE'],
    #                  'Average CAT score', survey_questions_replacement[2],
    #                  score_labels, answers_order[2],
    #                  # 'Relationship between CAT score and ease of use',
    #                  output_folders, 15)


def multiple_crosstab_heatmap(xs, y, x_labels, y_label, x_orders, y_order, output_directories, number):
    """
    Plot a cross-tabulation heatmap
    :param xs:                 the variables to use for the x axis
    :param y:                  the variable to use for the y axis
    :param x_labels:           the labels of the x axis for each variable
    :param y_label:            the label of the y axis
    :param x_orders:           the order of the x axis for each variable
    :param y_order:            the order of the y axis
    :param output_directories: the directories where to save the image
    :param number:             the number of the image
    :return:
    """
    h = 3.2
    w = h * len(xs)
    if len(y_order) > 3:
        w += len(y_order)
    fig, axs = plt.subplots(1, len(xs) + 1, figsize=(w, h),
                            # sharey=True,
                            gridspec_kw={'width_ratios': [10, 10, 10, 10, 1]}
                            )

    for i, x in enumerate(xs):
        x_label = x_labels[i]
        x_order = x_orders[i]

        crosstab = pd.crosstab(y, x, normalize=True) * 100

        # Reorder the crosstab
        crosstab = crosstab.reindex(columns=x_order, fill_value=0)
        crosstab = crosstab.reindex(y_order, fill_value=0)

        # Create a dict that maps the x-tick labels to new labels
        if x_label == 'Which resolution mode\ndid you prefer?':
            mapping = {
                'VPI-text': 'VPI\n(text)',
                'VPI-symbols': 'VPI\n(symbols)',
                'GI': 'GI'
            }
            x_order = [mapping[x] for x in x_order]

        cbar_flag = True if i == 3 else False
        ax_flag = True if i == 0 else False
        yticklabels = y_order if i == 0 else False

        # pos = axs[3].get_position()
        # cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])  # Adjust the values as needed

        res = sns.heatmap(ax=axs[i], data=crosstab, annot=True, fmt='.0f', cmap='Purples',
                          annot_kws={'size': 14}, vmin=0, vmax=100, cbar=cbar_flag, cbar_ax=axs[4],
                          yticklabels=yticklabels, xticklabels=x_order, #square=True, robust=True,
                          )
        res.set_xticklabels(res.get_xticklabels(), rotation=0)

        # Add the percentage symbol to the values of the heatmap
        for t in res.texts:
            t.set_text(t.get_text() + r'\%')

        # Invert y-axis
        axs[i].invert_xaxis()
        # fig.gca().invert_yaxis()
        # plt.gca().invert_yaxis()

        if y_label in ['Average CAT score', 'Average algorithm dimension']:
            axs[i].invert_yaxis()

        if ax_flag:
            axs[i].set_ylabel(y_label, labelpad=10)
        else:
            axs[i].set_ylabel('')

        axs[i].set_xlabel(x_label, labelpad=10)

    cbar = res.collections[0].colorbar
    #
    # # Set the colorbar ticks and labels as percentage
    cbar_ticks = np.arange(0, 101, 20)  # Ticks every 10%
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{tick}\%' for tick in np.arange(0, 101, 20)])

    fig.subplots_adjust(wspace=0.1)
    # plt.show()
    for out_dir in output_directories:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        extension = out_dir.split('/')[-2]

        plt.savefig(out_dir + 'multiple_crosstab{}.{}'.format(number, extension), dpi=600, transparent=True,
                    bbox_inches='tight', metadata={'Author': 'Giorgia Adorni'})

    plt.close()


def crosstab_heatmap(x, y, x_label, y_label, x_order, y_order, output_directories, number):
    """
    Plot a cross-tabulation heatmap
    :param x:                  the variable to use for the x axis
    :param y:                  the variable to use for the y axis
    :param x_label:            the label of the x axis
    :param y_label:            the label of the y axis
    :param x_order:            the order of the x axis
    :param y_order:            the order of the y axis
    :param output_directories: the directories where to save the image
    :param number:             the number of the image
    :return:
    """
    crosstab = pd.crosstab(x, y)

    # Reorder the crosstab
    crosstab = crosstab.reindex(columns=y_order, fill_value=0)
    crosstab = crosstab.reindex(x_order, fill_value=0)

    # Plot the cross-tabulation
    plt.figure(figsize=(6, 4))

    # Create a dict that maps the x-tick labels to new labels
    if y_label == 'Which resolution mode did you prefer?':
        mapping = {
            'VPI-text': 'VPI\n(text)',
            'VPI-symbols': 'VPI\n(symbols)',
            'GI': 'GI'
        }
        y_order = [mapping[x] for x in y_order]

    sns.heatmap(crosstab, annot=True, fmt="d", cmap='Purples',
                annot_kws={'size': 14}, vmin=0, vmax=100,
                yticklabels=x_order, xticklabels=y_order)

    # Invert y-axis
    plt.gca().invert_yaxis()

    plt.ylabel(x_label)
    plt.xlabel(y_label)

    for out_dir in output_directories:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        extension = out_dir.split('/')[-2]

        plt.savefig(out_dir + 'crosstab{}.{}'.format(number, extension), dpi=600, transparent=True,
                    bbox_inches='tight', metadata={'Author': 'Giorgia Adorni'})

    plt.close()


def responses_frequency(survey_data, survey_questions, survey_questions_replacement, answers_order, output_directories):
    """
    Plot the frequency of responses for each question
    :param survey_data:                  the DataFrame containing the survey data
    :param survey_questions:             the questions of the survey
    :param survey_questions_replacement: the replacement for the questions of the survey
    :param answers_order:                the order of the answers for each question
    :param output_directories:           the directories where to save the plots
    """
    # Define a list of colours to cycle through, green yellow and red
    colours = ['tab:green', '#FFD700', 'tab:red']

    plt.figure(figsize=(14, 6))

    ax = plt.subplot(111)

    bottom = None

    new_survey_data = survey_data.copy()

    # Remove the first 3 columns from new_survey_data
    new_survey_data = new_survey_data.drop(new_survey_data.columns[[0, 1, 2]], axis=1)

    # Replace in new_survey_data the i-th answer in answers_order with a new label
    new_answers = ['Positive', 'Neutral', 'Negative']
    for i, question in enumerate(survey_questions, 1):
        # If the answer corresponds to the element at index 0 in the answers_order list, map it to 'Positive',
        # if it corresponds to the element at index 1 in the answers_order list, map it to 'Neutral',
        # if it corresponds to the element at index 2 in the answers_order list, map it to 'Negative'
        new_survey_data[question] = new_survey_data[question].map({answers_order[i - 1][0]: new_answers[0],
                                                                   answers_order[i - 1][1]: new_answers[1],
                                                                   answers_order[i - 1][2]: new_answers[2]})

    stacked_data = new_survey_data.apply(pd.Series.value_counts).fillna(0).T
    # Reorder the columns of stacked_data to be consistent with the order in new_answers
    stacked_data = stacked_data[new_answers]

    for i, col in enumerate(stacked_data.columns):
        plt.barh(y=survey_questions_replacement, width=stacked_data[col], left=bottom, color=colours[i])
        bottom = stacked_data[col] if bottom is None else bottom + stacked_data[col]

    plt.xlim(0, 129)
    ax.set_xticks(range(0, 131, 10))
    plt.xlabel('Count')

    plt.gca().invert_yaxis()

    legend_handles = []
    for i, question in enumerate(survey_questions_replacement):
        legend_handles += [mlines.Line2D([], [], color=colour, marker='s', markersize=10,
                                         linestyle='None', label=answers_order[i][j])
                           for j, colour in enumerate(colours)]

    # Reshape the legend handles list to have 3 columns
    handles_array = np.array(legend_handles)
    n_col = 3
    n_row = int(np.ceil(len(handles_array) / n_col))
    handles_reshaped = handles_array.reshape(n_row, n_col).T.flatten().tolist()

    plt.legend(handles=handles_reshaped,
               bbox_to_anchor=(1.05, .95), loc='upper left',
               ncol=n_col, columnspacing=1, handlelength=1.2, handletextpad=0.5, labelspacing=2.45)

    plt.tight_layout()

    for out_dir in output_directories:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        extension = out_dir.split('/')[-2]

        plt.savefig(out_dir + 'responses_frequency.{}'.format(extension), dpi=600, transparent=True,
                    bbox_inches='tight', metadata={'Author': 'Giorgia Adorni'})

    plt.close()

def responses_frequency_hue(survey_data, survey_questions, survey_questions_replacement, answers_order,
                            typology, hue, output_directories):
    """
    Plot the frequency of responses for each question
    :param survey_data:                  the DataFrame containing the survey data
    :param survey_questions:             the questions of the survey
    :param survey_questions_replacement: the replacement for the questions of the survey
    :param answers_order:                the order of the answers for each question
    :param typology:                     the type of plot
    :param hue:                          the column to use for grouping the data
    :param output_directories:           the directories where to save the plots
    """
    # Define a list of colours to cycle through, green yellow and red
    colours = ['tab:green', '#FFD700', 'tab:red']

    plt.figure(figsize=(4*4, 4*2))

    # Plot the frequency of responses for each question
    for i, question in enumerate(survey_questions, 1):
        plt.subplot(2, 4, i)

        crosstab = pd.crosstab(survey_data[hue], survey_data[question])

        if hue == 'AGE_GROUP':
            sorted_index = sorted(crosstab.index, key=lambda x: int(x.split('-')[0]))
            crosstab = crosstab.reindex(sorted_index)

        # Rescale it
        crosstab = crosstab.div(crosstab.sum(axis=1), axis=0)

        plt.ylim(0, 1 + 0.01)
        yticks_numbers = np.arange(0, 1 + 0.01, 0.2)

        # Use a custom formatter to append the % symbol
        formatter = mticker.FuncFormatter(lambda x, _: f'{int(x * 100)}\%')
        plt.gca().yaxis.set_major_formatter(formatter)

        # Set the y-ticks
        plt.yticks(yticks_numbers)
        plt.ylabel('Percentage')

        bottom = None
        for answer in answers_order[i - 1]:
            plt.bar(crosstab.index, crosstab[answer], bottom=bottom, label=answer,
                    color=colours[answers_order[i - 1].index(answer)], width=0.7)
            bottom = crosstab[answer] if bottom is None else bottom + crosstab[answer]

        plt.grid(axis='y')
        plt.xlabel(typology)

        ax = plt.gca()  # Get the current axis

        ax.set_xticks(range(len(crosstab.index)))
        ax.set_xticklabels(crosstab.index)

        ax.get_xticklabels()[0].set_y(0.005)

        if survey_questions_replacement[i - 1] == 'Which resolution mode did you prefer?':
            # Change the legend labels
            labels = ['VPI\n(text)', 'VPI\n(symbols)', 'GI']
            handles, _ = ax.get_legend_handles_labels()
            leg = ax.legend(handles, labels, title=survey_questions_replacement[i - 1], bbox_to_anchor=(0.5, 1.45),
                            loc='upper center', ncol=3, handlelength=1, handletextpad=0.5, handleheight=1,
                            columnspacing=1)
        # elif survey_questions_replacement[i - 1] == 'Did you enjoy this activity?':
        #     # Change the legend labels
        #     labels = ['Definitely', 'So-so', 'No']
        #     handles, _ = ax.get_legend_handles_labels()
        #     leg = ax.legend(handles, labels, title=survey_questions_replacement[i - 1], bbox_to_anchor=(0.5, 1.45),
        #                     loc='upper center', ncol=3, handlelength=1, handletextpad=0.5, handleheight=1,
        #                     columnspacing=1)
        # elif survey_questions_replacement[i - 1] == 'Have you ever used an app like this\nto do exercises and learn?':
        #     # Change the legend labels
        #     labels = ['Yes', "Don't recall", 'Never']
        #     handles, _ = ax.get_legend_handles_labels()
        #     leg = ax.legend(handles, labels, title=survey_questions_replacement[i - 1], bbox_to_anchor=(0.5, 1.45),
        #                     loc='upper center', ncol=3, handlelength=1, handletextpad=0.5, handleheight=1,
        #                     columnspacing=1)
        # elif survey_questions_replacement[i - 1] == 'Would you do this experience again?':
        #     # Change the legend labels
        #     labels = ['Yes', 'Maybe', 'No']
        #     handles, _ = ax.get_legend_handles_labels()
        #     leg = ax.legend(handles, labels, title=survey_questions_replacement[i - 1], bbox_to_anchor=(0.5, 1.45),
        #                     loc='upper center', ncol=3, handlelength=1, handletextpad=0.5, handleheight=1,
        #                     columnspacing=1)
        else:
            leg = ax.legend(title=survey_questions_replacement[i - 1], bbox_to_anchor=(0.5, 1.45), loc='upper center',
                            ncol=3, handlelength=1, handletextpad=0.5, handleheight=1, columnspacing=1)

        leg.get_title().set_multialignment('center')

        # Adjust the layout of the figure to fit the outside legend
        plt.subplots_adjust(hspace=0.7, wspace=0.6)

    for out_dir in output_directories:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        extension = out_dir.split('/')[-2]

        plt.savefig(out_dir + 'responses_frequency_by_{}.{}'.format(typology, extension), dpi=600, transparent=True,
                    bbox_inches='tight', metadata={'Author': 'Giorgia Adorni'})

    plt.close()


def get_score_data_and_annotations(dataframe, gender):
    """
    Get the data and annotations for the score by age category plot
    :param dataframe: the DataFrame containing the data
    :param gender:    the gender to filter the data
    :return:          the data and annotations for the score by age category plot
    """
    # Compute the percentage of students w.r.t. age category
    score_df = dataframe.loc[dataframe['GENDER'] == gender] if gender else dataframe.copy()
    score_df = score_df.groupby(['AGE_CATEGORY', 'CAT_SCORE'], observed=False).count()[['AGE_GROUP']].reset_index()
    score_df = score_df.pivot(index='AGE_CATEGORY', columns='CAT_SCORE', values='AGE_GROUP').transpose().fillna(0)

    # Check if rows 0 to 5 are present in the index
    expected_rows = pd.Index(range(5))
    missing_rows = expected_rows[~expected_rows.isin(score_df.index)]

    if not missing_rows.empty:
        # Add missing rows with NaN values and fill NaNs with 0
        missing_df = pd.DataFrame(index=missing_rows, columns=score_df.columns)
        score_df = pd.concat([score_df, missing_df]).sort_index().fillna(0)

    data = score_df / score_df.sum() * 100
    annot = score_df.astype(int).astype(str) + '\n (' + data.round().astype(int).astype(str) + '\%)'

    # For column 'From 3 to 6 years old' in data, replace rows 4.0 and 5.0 with np.nan
    data.loc[4.0, 'From 3 to 6 years old'] = np.nan
    data.loc[5.0, 'From 3 to 6 years old'] = np.nan

    # For column 'From 3 to 6 years old' in annot, replace rows 4.0 and 5.0 with -
    annot.loc[4.0, 'From 3 to 6 years old'] = ''
    annot.loc[5.0, 'From 3 to 6 years old'] = ''

    return annot, data


def get_algorithm_and_artefact_dimensions(dataframe):
    """
    Get the unique values of the columns ALGORITHM_DIMENSION and ARTEFACT_DIMENSION
    :param dataframe: the DataFrame containing the data
    :return:          the unique values of the columns ALGORITHM_DIMENSION and ARTEFACT_DIMENSION
    """
    # Get the unique values of the column AGE_CATEGORY and AGE_CATEGORY_DESCRIPTION
    age_category_names = dataframe.groupby('AGE_GROUP')['AGE_CATEGORY'].unique().explode().unique()

    # Get the unique values of the columns ALGORITHM_DIMENSION and ARTEFACT_DIMENSION
    algorithm = dataframe.groupby('ALGORITHM_DIMENSION')['ALGORITHM_TYPE'].unique().explode().unique()
    artefact = dataframe.groupby('ARTEFACT_DIMENSION')['ARTEFACT_TYPE'].unique().explode().unique()

    return age_category_names, algorithm, artefact


def heatmap_algorithm_vs_artefact_dimension(age_category_names, algorithm, artefact, axs, combined_df_crosstab,
                                            filtered_dataframe, fig, plot_type, k=0):
    """
    Plot the heatmap of the algorithm dimension vs artefact dimension for each age category.
    :param age_category_names:   the unique values of the column AGE_CATEGORY and AGE_CATEGORY_DESCRIPTION
    :param algorithm:            the unique values of the columns ALGORITHM_DIMENSION and ALGORITHM_TYPE
    :param artefact:             the unique values of the columns ARTEFACT_DIMENSION and ARTEFACT_TYPE
    :param axs:                  the axes of the plot
    :param combined_df_crosstab: the crosstab of the combined DataFrame
    :param filtered_dataframe:   the filtered DataFrame
    :param fig:                  the figure of the plot
    :param plot_type:            the type of plot
    :param k:                    the increase factor for the index of the subplot
    :return:                     the plot
    """
    for idx, age_category in enumerate(age_category_names):
        # Get the DataFrame with the data of the current age category
        age_df = filtered_dataframe[filtered_dataframe['AGE_CATEGORY'] == age_category]

        # Get the reduced DataFrame with only the columns ALGORITHM_TYPE and ARTEFACT_TYPE
        reduced_df = age_df.filter(['ALGORITHM_TYPE', 'ARTEFACT_TYPE'])

        # Get the crosstab of the reduced DataFrame
        cross_tab = pd.crosstab(reduced_df['ALGORITHM_TYPE'], reduced_df['ARTEFACT_TYPE'], normalize=True)

        # Get the full DataFrame with the crosstab of the combined DataFrame
        full_df = cross_tab.join(combined_df_crosstab, lsuffix='_', rsuffix='_', how='outer')
        full_df = full_df.T.groupby(level=0).sum().T
        full_df = full_df.rename(columns={'GF_': 'GF', 'G_': 'G',
                                          'PF_': 'PF', 'P_': 'P'})
        full_df = full_df[artefact]
        full_df = full_df.reindex(algorithm[::-1])
        full_df = full_df * 100

        # Replace the 0 values for PF and P with NaN if the age category is 'From 3 to 6 years old'
        if age_category == 'From 3 to 6 years old':
            full_df['PF'] = full_df['PF'].replace(0, np.nan)
            full_df['P'] = full_df['P'].replace(0, np.nan)

        print('Age category: {}'.format(age_category))
        print(full_df.to_string())

        # Set the title of the plot
        axs[idx + k].set_title(r'\textbf{{{}}}'.format(age_category_names[idx]), fontsize=15, pad=30)

        # Create the seaborn heatmap
        res = sns.heatmap(ax=axs[idx + k], data=full_df, cmap='Greens', annot=True, vmin=0, vmax=100, cbar=False,
                          fmt='.0f', square=True, annot_kws={'size': 14})

        if plot_type == 'gender':
            fig.subplots_adjust(top=0.95, wspace=0.55, hspace=0.7)
        elif plot_type == 'schema':
            fig.subplots_adjust(wspace=0.6)
        else:
            fig.subplots_adjust(wspace=0.55)

        if age_category == 'From 3 to 6 years old':
            nrows, ncols = full_df.shape

            # Loop over cells and add hatches
            for i in range(nrows):
                for j in range(ncols):
                    # Verify if the value is NaN
                    if pd.isna(full_df.iloc[i, j]):
                        # If the value is NaN, add a hatch to the cell
                        hatch_rect = plt.Rectangle((j, i), 1, 1, fill=True, hatch='//',
                                                   edgecolor='gray', facecolor='none', linewidth=0)
                        axs[idx + k].add_patch(hatch_rect)

        # Add the percentage symbol to the values of the heatmap
        for t in res.texts:
            t.set_text(t.get_text() + r'\%')

        # Compute the algorithm dimension total percentage
        row_percentages = full_df.sum(axis=1)
        annot_rows = row_percentages.round().astype(int).astype(str) + r'\%'

        # Add it percentage to the plot
        nrows = full_df.shape[0]
        for i in range(nrows):
            # Compute the y position of the text and add 0.5 to center it in the row
            y_pos = i + 0.5
            axs[idx + k].text(axs[idx + k].get_xlim()[1] + 0.35, y_pos, annot_rows.iloc[i], size=14, ha='center',
                              va='center')

        # Compute the interaction dimension total percentage
        columns_percentages = full_df.sum()
        annot_columns = columns_percentages.round().astype(int).astype(str) + r'\%'

        # If the age category is 'From 3 to 6 years old', replace the annotations for columns 'PF' and 'P' with '-'
        if age_category == 'From 3 to 6 years old':
            annot_columns['PF'] = '-'
            annot_columns['P'] = '-'

        # Add it to the plot
        ncols = full_df.shape[1]
        for j in range(ncols):
            # Compute the x position of the text and add 0.5 to center it in the column
            x_pos = j + 0.5
            axs[idx + k].text(x_pos, axs[idx + k].get_ylim()[1] - 0.2, annot_columns.iloc[j], size=14, ha='center',
                              va='center')

        # Set the ticks
        axs[idx + k].tick_params(left=True, bottom=True)

        # Set the labels
        axs[idx + k].set_xlabel('Interaction dimension')
        axs[idx + k].set_ylabel('Algorithm dimension')


def performance_by_age_category(df, output_directories):
    """
    Plot the performance of the students, in terms of algorithm dimension and artefact dimension by age category.
    :param df:                 the DataFrame containing the data
    :param output_directories: the directories where to save the plots
    :return:                   the plot
    """
    dataframe = df.copy()

    # Get the unique values of the columns ALGORITHM_DIMENSION and ARTEFACT_DIMENSION
    age_category_names, algorithm, artefact = get_algorithm_and_artefact_dimensions(dataframe)

    # Create a DataFrame with all the possible combinations of algorithm dimension and artefact dimension
    combined_df_crosstab = get_combined_crosstab(algorithm, artefact)

    columns_to_keep = ['ALGORITHM_TYPE', 'AGE_CATEGORY', 'ARTEFACT_TYPE', 'GENDER']
    dataframe = dataframe[columns_to_keep]

    # Plots with no gender
    gender = ''

    # Get the DataFrame with the data of the selected gender
    filtered_df = dataframe.loc[dataframe['GENDER'] == gender] if gender else dataframe.copy()

    h = 4
    w = h * len(age_category_names)
    fig, axs = plt.subplots(1, len(age_category_names), figsize=(w, h))

    heatmap_algorithm_vs_artefact_dimension(age_category_names, algorithm, artefact, axs, combined_df_crosstab,
                                            filtered_df, fig, plot_type='')

    # Save the plot
    for out_dir in output_directories:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        extension = out_dir.split('/')[-2]

        plt.savefig(out_dir + 'performance_by_age_category.{}'.format(extension), dpi=600, transparent=True,
                    bbox_inches='tight', metadata={'Author': 'Giorgia Adorni'})

    plt.close()

    # Dual plot for both genders
    fig, axs = plt.subplots(2, len(age_category_names), figsize=(w, h + h - 1))

    center_x = 0.5
    y_positions = [1.05, 0.525]

    for i, gender in enumerate(['Male', 'Female']):
        # Get the DataFrame with the data of the selected gender
        filtered_df = dataframe.loc[dataframe['GENDER'] == gender] if gender else dataframe.copy()

        heatmap_algorithm_vs_artefact_dimension(age_category_names, algorithm, artefact, axs[i, :],
                                                combined_df_crosstab, filtered_df, fig, plot_type='gender')

        fig.text(center_x, y_positions[i], r'\textbf{{{}}}'.format(gender), fontsize=18, ha='center', va='center')

    # Save the plot
    for out_dir in output_directories:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        extension = out_dir.split('/')[-2]

        plt.savefig(out_dir + 'performance_by_age_category_genders.{}'.format(extension), dpi=600, transparent=True,
                    bbox_inches='tight', metadata={'Author': 'Giorgia Adorni'})

    plt.close()

    print('Saved plot performance_by_age_category_{}'.format(gender))


def get_combined_crosstab(algorithm, artefact):
    # Create a DataFrame with all the possible combinations of algorithm dimension and artefact dimension
    combined = [algorithm, artefact]
    combined_df = pd.DataFrame(columns=['ALGORITHM_TYPE', 'ARTEFACT_TYPE'],
                               data=list(itertools.product(*combined)))
    combined_df_crosstab = pd.crosstab(combined_df['ALGORITHM_TYPE'], combined_df['ARTEFACT_TYPE'])
    combined_df_crosstab = 1 - combined_df_crosstab
    return combined_df_crosstab


def score_by_age_category(df, output_directories):
    """
    Plot the performance of the students, in terms of CAT score and age category
    :param df:                 the DataFrame containing the data
    :param output_directories: the directories where to save the plots
    :return:                   the plot
    """
    dataframe = df.copy()
    age_category_order = dataframe.groupby('AGE_GROUP')[
        'AGE_CATEGORY'].unique().explode().unique().tolist()  # Update with the desired order of AGE_GROUP

    # Convert SCHEMA_ID and AGE_GROUP to categorical data types with the specified order
    dataframe['AGE_CATEGORY'] = pd.Categorical(dataframe['AGE_CATEGORY'], categories=age_category_order, ordered=True)

    age_category_names, algorithm, artefact = get_algorithm_and_artefact_dimensions(dataframe)

    columns_to_keep = ['CAT_SCORE', 'AGE_CATEGORY', 'AGE_GROUP', 'GENDER']
    dataframe = dataframe[columns_to_keep]

    # Plots with no gender
    gender = ''

    annot, data = get_score_data_and_annotations(dataframe, gender)

    print(gender)
    print(data.to_string())

    # Set figure size
    h = data.shape[0] + 1

    plt.figure(figsize=(h, h))
    ax = plt.gca()
    sns.heatmap(data, annot=annot, fmt='', cmap='Blues', annot_kws={'size': 14},
                cbar_kws={'label': 'Percentage of students w.r.t age category \n (column-wise normalisation)'},
                vmax=100, vmin=0, ax=ax)

    nrows, ncols = data.shape

    # Loop over cells and add hatches
    for i in range(nrows):
        for j in range(ncols):
            # Verify if the value is NaN
            if pd.isna(data.iloc[i, j]):
                # If the value is NaN, add a hatch to the cell
                hatch_rect = plt.Rectangle((j, i), 1, 1, fill=True, hatch='//',
                                           edgecolor='gray', facecolor='none', linewidth=0)
                ax.add_patch(hatch_rect)

    ax.set_xlim(0, len(age_category_names))
    ax.set_ylim(0, 6)

    new_age_category_names = ['3-6 yrs', '7-9 yrs', '10-13 yrs', '14-16 yrs']
    ax.set_xticklabels([r'{{{}}}'.format(age_category) for age_category in new_age_category_names], rotation=0)

    score_names = ['0', '1', '2', '3', '4', '5']
    ax.set_yticklabels([r'{{{}}}'.format(score) for score in score_names])

    ax.set_xlabel('Age category')
    ax.set_ylabel('CAT score')

    ax.tick_params(left=True, bottom=True)

    for out_dir in output_directories:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        extension = out_dir.split('/')[-2]

        plt.savefig(out_dir + 'score_by_age_category.{}'.format(extension), dpi=600, transparent=True,
                    bbox_inches='tight', metadata={'Author': 'Giorgia Adorni'})

    plt.close()
    print('Saved plot score_by_age_category')

    # Dual plot for both genders
    fig, axs = plt.subplots(1, 2, figsize=(h * 1.5, h), sharey=True)

    # Get the position of the first plot
    pos = axs[1].get_position()

    # Create a new axes for the colorbar
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])  # Adjust the values as needed

    for i, gender in enumerate(['Male', 'Female']):
        annot, data = get_score_data_and_annotations(dataframe, gender)

        print(gender)
        print(data.to_string())

        ax = axs[i]
        sns.heatmap(data, annot=annot, fmt='', cmap='Blues', annot_kws={'size': 14},
                    cbar_kws={'label': 'Percentage of students w.r.t age category \n (column-wise normalisation)'},
                    vmax=100, vmin=0, ax=ax, cbar=i == 0, cbar_ax=cbar_ax)

        nrows, ncols = data.shape

        # Loop over cells and add hatches
        for i in range(nrows):
            for j in range(ncols):
                # Verify if the value is NaN
                if pd.isna(data.iloc[i, j]):
                    # If the value is NaN, add a hatch to the cell
                    hatch_rect = plt.Rectangle((j, i), 1, 1, fill=True, hatch='//',
                                               edgecolor='gray', facecolor='none', linewidth=0)
                    ax.add_patch(hatch_rect)

        ax.set_xlim(0, len(age_category_names))
        ax.set_ylim(0, 6)

        ax.set_xticklabels([r'{{{}}}'.format(age_category) for age_category in age_category_names], rotation=0)
        ax.set_xlabel('Age category')

        if i == 0:
            ax.set_ylabel('CAT score')  # Set y-label only for the first subplot
            ax.tick_params(left=True, bottom=True)
        else:
            ax.set_ylabel('')
            ax.tick_params(bottom=True)

        ax.set_title(r'\textbf{{{}}}'.format('Male' if gender == 'Male' else 'Female'), fontsize=18, pad=20)

    fig.subplots_adjust(wspace=0.05)

    # Save the plot
    for out_dir in output_directories:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        extension = out_dir.split('/')[-2]

        plt.savefig(out_dir + 'score_by_age_category_genders.{}'.format(extension), dpi=600, transparent=True,
                    bbox_inches='tight', metadata={'Author': 'Giorgia Adorni'})

    plt.close()

    print('Saved plot score_by_age_category_genders')


def performance_by_age_category_and_schema(df, sequence_folder, output_directories):
    """
    Plot the performance of the students, in terms of CAT score and age category, for each schema
    :param df:                 the DataFrame containing the data
    :param sequence_folder:    the folder containing the images of the schemas
    :param output_directories: the directories where to save the plots
    :return:                   the plot
    """
    dataframe = df.copy()

    # Get the paths of the images and the names of the schemas
    images, schemas = import_png_images(sequence_folder)

    # Get the unique values of the columns ALGORITHM_DIMENSION and ARTEFACT_DIMENSION
    age_category_names, algorithm, artefact = get_algorithm_and_artefact_dimensions(dataframe)

    # Create a DataFrame with all the possible combinations of algorithm dimension and artefact dimension
    combined_df_crosstab = get_combined_crosstab(algorithm, artefact)

    columns_to_keep = ['ALGORITHM_TYPE', 'AGE_CATEGORY', 'ARTEFACT_TYPE', 'SCHEMA_NAME']
    dataframe = dataframe[columns_to_keep]

    for idx, schema in enumerate(schemas):
        # Get the data for the current schema
        filtered_df = dataframe[dataframe['SCHEMA_NAME'] == schema]

        h = 4
        w = h * len(age_category_names) + h

        fig, axs = plt.subplots(1, len(age_category_names) + 1,
                                figsize=(w, h))

        # Add the schema to the first axis
        img = mpimg.imread(images[idx])
        axs[0].imshow(img)
        axs[0].axis('off')

        heatmap_algorithm_vs_artefact_dimension(age_category_names, algorithm, artefact, axs, combined_df_crosstab,
                                                filtered_df, fig, k=1, plot_type='schema')

        # Save the plot
        for out_dir in output_directories:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            extension = out_dir.split('/')[-2]

            plt.savefig(out_dir + 'performance_by_age_category_and_schema_{}.{}'.format(schema, extension),
                        dpi=600, transparent=True, bbox_inches='tight', metadata={'Author': 'Giorgia Adorni'})

        plt.close()

        print('Saved plot performance_by_age_category_and_schema_{}'.format(schema))


def artefacts_by_age_category(df, output_directories):
    """
    Generate the LaTeX table for the artefacts by age category analysis
    :param df:                 the DataFrame containing the data
    :param output_directories: the directories where to save the plots
    :return:                   the plot
    """
    dataframe = df.copy()

    type = 'ARTEFACT_TYPE'
    dimension = 'ARTEFACT_DIMENSION'

    predominant_type = 'PREDOMINANT_ARTEFACT_TYPE'
    predominant_dimension = 'PREDOMINANT_ARTEFACT_DIMENSION'

    # Set the order of the categories
    artefact_category_order = dataframe.groupby(dimension)[type].unique().explode().unique().tolist()
    dataframe[type] = pd.Categorical(dataframe[type], categories=artefact_category_order, ordered=True)

    age_category_order = dataframe.groupby('AGE_GROUP')['AGE_CATEGORY'].unique().explode().unique().tolist()
    dataframe['AGE_CATEGORY'] = pd.Categorical(dataframe['AGE_CATEGORY'],
                                               categories=age_category_order,
                                               ordered=True)

    # Group by AGE_CATEGORY and ARTEFACT_CATEGORY and calculate count
    grouped = dataframe.groupby(['AGE_CATEGORY', type], observed=False).size().reset_index(name='COUNT')

    # Pivot the result to create the analysis table
    np_analysis_table = grouped.pivot(index='AGE_CATEGORY', columns=type, values='COUNT').fillna(0)

    # Set the order of the categories for the predominant artefact type
    predominant_artefact_category_order = dataframe.groupby(predominant_dimension)[predominant_type].unique().explode().unique().tolist()
    dataframe[predominant_type] = pd.Categorical(dataframe[predominant_type],
                                                 categories=predominant_artefact_category_order, ordered=True)

    predominant_grouped = dataframe.groupby(['AGE_CATEGORY', predominant_type],
                                            observed=False).size().reset_index(name='COUNT')

    p_analysis_table = predominant_grouped.pivot(index='AGE_CATEGORY',
                                                 columns=predominant_type, values='COUNT').fillna(0)

    # Create a stacked bar plot using Seaborn
    plt.figure(figsize=(7, 4))
    ax = plt.subplot(111)

    cmap = plt.get_cmap('Reds_r')

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    new_cmap = truncate_colormap(cmap, 0.2, 0.8)

    bar_width = 0.4
    x_labels = ['1', '2', '3', '4']
    indices = np.arange(len(x_labels))

    num_layers = len(np_analysis_table.columns)
    hatch_patterns = ['', '//']

    # Plot the stacked bars
    for df, offset, hatch in [(np_analysis_table, -bar_width / 2, hatch_patterns[0]), (p_analysis_table, bar_width / 2, hatch_patterns[1])]:
        bottom = np.zeros(len(df))
        for i, col in enumerate(df.columns):
            # Get the color for the current layer
            colour = new_cmap(i / num_layers)

            ax.bar(indices + offset, df[col], bar_width, bottom=bottom, color=colour,
                   label=f'{col} ({i + 1})' if bottom.sum() == 0 else "", hatch=hatch)

            bottom += df[col].values

    plt.xlabel('Age group')
    plt.ylabel('Count')

    # Customize the plot
    sns.despine(left=True, bottom=True)  # Remove background

    ax.set_xticks(indices)
    ax.set_xticklabels(x_labels)

    ax.set_ylim(0, 450)
    ax.yaxis.grid()  # Add horizontal grid lines

    # Interaction Dimensions Legend
    # interaction_legend = ['GF', 'G', 'PF', 'P']

    plt.text(x=1.0, y=0.82, s='Lowest', transform=ax.transAxes, fontsize=12, color='black', zorder=3)
    plt.text(x=1.22, y=0.82, s='Prevalent', transform=ax.transAxes, fontsize=12, color='black', zorder=3)

    # # Create handles using the colors of the stacked bars
    # handles_np = [Patch(color='none', label='Actual')]
    custom_handle = mlines.Line2D([], [], color='none', marker='s', markersize=3,
                                  label='', linestyle='None')
    handles_np = ([custom_handle] +
                  [Patch(color=new_cmap(i / num_layers), label=f'{col}') for i, col in
                   enumerate(np_analysis_table.columns)])

    handles_p = ([Patch(color='none', label='')] +
                 [Patch(facecolor=new_cmap(i / num_layers), label=f'{col}', edgecolor='white',
                        hatch=hatch_patterns[1]) for i, col in enumerate(np_analysis_table.columns)])

    handles = handles_np + handles_p

    interaction_legend = ax.legend(handles=handles, bbox_to_anchor=(.95, 1), loc='upper left',
                                   ncol=2, columnspacing=1.1, handlelength=2)
    # interaction_legend.get_frame().set_facecolor('none')
    interaction_legend.set_title('Interaction dimension')
    # interaction_legend.get_title().set_multialignment('center')

    interaction_legend.set_zorder(2)

    # Add the age category mapping to the legend
    age_category_legend = [
        ('1', '3-6 yrs'),
        ('2', '7-9 yrs'),
        ('3', '10-13 yrs'),
        ('4', '14-16 yrs')
    ]
    age_category_handles = [Patch(color='none', label=f'{group}: {category}') for group, category in
                            age_category_legend]
    age_legend = ax.legend(handles=age_category_handles, bbox_to_anchor=(1.06, 0),
                           loc='lower left', borderaxespad=0., handlelength=0, handletextpad=0)
    age_legend.set_title('Age category')
    # age_legend.get_title().set_multialignment('center')

    # Add the interaction dimensions legend back to the plot
    plt.gca().add_artist(interaction_legend)

    interaction_legend.set_zorder(2)

    plt.tight_layout()

    # Save the plot
    for out_dir in output_directories:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        extension = out_dir.split('/')[-2]

        plt.savefig(out_dir + 'artefact_by_age_group.{}'.format(extension), dpi=600, transparent=True,
                    bbox_inches='tight', metadata={'Author': 'Giorgia Adorni'}, bbox_extra_artists=(age_legend, interaction_legend))

    plt.close()

    print('Saved plot artefact_by_age')
