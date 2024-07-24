"""
@file-name:     data_analysis.py
@date-creation: 02.09.2023
@author-name:   Giorgia Adorni
"""
import argparse
import os

import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from plots import (performance_by_age_category, score_by_age_category, performance_by_age_category_and_schema,
                   artefacts_by_age_category, generate_restart_distribution_plot, plot_interaction_dimension,
                   responses_frequency, plot_crosstabs, plot_group_frequencies, plot_score_by_schema_and_group,
                   plot_score_by_group, generate_time_plot)
from tables import (time_by_schema_latex_table, performance_by_category_latex_table,
                    algorithms_by_age_latex_table, performance_by_schema_latex_table, algorithms_by_schema_latex_table,
                    algorithms_latex_table, time_by_category_latex_table, performance_by_schema_and_age_latex_table,
                    students_by_school_latex_table, students_latex_table)
from utils import import_tables, format_time, print_df_info


def analyse_correlations(df):
    """
    Analyse correlations
    :param df: the DataFrame containing the data
    """
    dataframe = df.copy()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Set the display option to display all the text in a cell
    pd.set_option('display.max_colwidth', None)

    # Set the display option to display all the columns
    pd.set_option('display.width', None)

    # Calculate the correlation matrix using columns
    # ARTEFACT_DIMENSION, ALGORITHM_DIMENSION, ARMOS_GRADE, CLASS,
    # SCHOOL_ID, AGE, AGE_GROUP, CAT_SCORE, GENDER_IDX, CANTON_ID
    corr_matrix = dataframe.corr()
    print(corr_matrix)

    # Plot the correlation matrix
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.show()


def generate_plots(df, schemas_folder, output_folders):
    """
    Generate plots
    :param df:             the DataFrame containing the data
    :param schemas_folder: the folder containing the schemas
    :param output_folders: the folder where to save the images
    """
    # Plot the performance of the students, in terms of algorithm dimension and artefact dimension by age group
    print('\nPlotting performance by age category')
    performance_by_age_category(df, output_folders)

    # Plot the score of the students, in terms of CAT score and age category
    print('\nPlotting score by age category')
    score_by_age_category(df, output_folders)

    # Plot, for each schema, the performance of the students,
    # in terms of algorithm dimension and interaction dimension by age group
    print('\nPlotting performance by age category and schema')
    performance_by_age_category_and_schema(df, schemas_folder, output_folders)

    # Plot the distribution of artefact used by age categoryu

    artefacts_by_age_category(df, output_folders)

    plot_score_by_group(df, 'CAT_SCORE', 'Avg CAT score',
                        'AGE_CATEGORY', output_folders)

    plot_score_by_group(df, 'ALGORITHM_DIMENSION', 'Avg algorithm dimension',
                        'AGE_CATEGORY', output_folders)
    #
    plot_score_by_schema_and_group(df, 'CAT_SCORE', 'Avg CAT score',
                                   'AGE_CATEGORY', output_folders)

    plot_score_by_schema_and_group(df, 'ALGORITHM_DIMENSION', 'Avg algorithm dimension',
                                   'AGE_CATEGORY', output_folders)

    # NEW WEIGHTED SCORES
    plot_score_by_group(df, 'WEIGHTED_CAT_SCORE', 'Adapted avg CAT score',
                        'AGE_CATEGORY', output_folders)

    plot_score_by_group(df, 'WEIGHTED_ALGORITHM_DIMENSION', 'Adapted avg algorithm dimension',
                        'AGE_CATEGORY', output_folders)

    plot_score_by_schema_and_group(df, 'WEIGHTED_CAT_SCORE', 'Adapted avg CAT score',
                                   'AGE_CATEGORY', output_folders)

    plot_score_by_schema_and_group(df, 'WEIGHTED_ALGORITHM_DIMENSION', 'Adapted avg algorithm dimension',
                                   'AGE_CATEGORY', output_folders)


def generate_paper_tables(df, tables_output_folder):
    """
    Generate paper tables
    :param df:                   the DataFrame containing the data
    :param tables_output_folder: the folder where to save the tables
    """
    # Generate a LaTeX table with the results of the student analysis
    print('\nGenerating LaTeX table with the results of the student analysis')
    students_latex_table(df,
                         'Demographic analysis of students by school type and age category. '
                         'The table provides an overview of the gender distribution among students in different '
                         'school types and age category. '
                         'The Female and Male columns represent the number of female and male students, respectively, '
                         'while the Total column displays the combined count. '
                         'The mean age ($\mu$) and standard deviation ($\pm$) are presented for each age category.',
                         'tab:students_analysis',
                         tables_output_folder)

    students_by_school_latex_table(df,
                                   'Demographic analysis of students by session. '
                                   'The table provides an overview of student demographics, including school type, '
                                   'canton, age category, and gender distribution among students in different sessions. '
                                   'The Female and Male columns represent the number of female and male students, respectively, '
                                   'while the Total column displays the combined count. '
                                   'The mean age ($\mu$) and standard deviation ($\pm$) are presented for each age category.',
                                   'tab:students_analysis_by_school',
                                   tables_output_folder)

    # Generate a LaTeX table with the results of the performance analysis by schema
    print('\nGenerating LaTeX table with the results of the performance analysis across schemas')
    performance_by_schema_latex_table(df,
                                      'Analysis of student performance across schemas. '
                                      'The table presents the number and percentage of students who attempted '
                                      'and solved each schema.'
                                      'The percentage of ``solved'' schemas is calculated only among pupils who attempted the schema.',
                                      'tab:performance_by_schema',
                                      tables_output_folder)

    # Generate a LaTeX table with the results of the performance analysis by age category and schema
    print('\nGenerating LaTeX table with the results of the performance analysis across age categories and schemas')
    performance_by_schema_and_age_latex_table(df, 'Analysis of student performance across age categories and schemas. '
                                                  'The table presents the number and percentage of students who attempted '
                                                  'and solved each schema, for each age category. '
                                                  'The percentage of ``solved'' schemas is calculated only among pupils who attempted the schema.',
                                              'tab:performance_by_schema_and_age',
                                              tables_output_folder)

    # Generate a LaTeX table with the results of the time analysis by age category
    print('\nGenerating LaTeX table with the results of the time analysis across age categories')
    time_by_category_latex_table(df,
                                 'AGE',
                                 'Analysis of activity completion time across age categories. '
                                 'The table presents a comprehensive overview of the time students of various '
                                 'age category take to complete all schemas. '
                                 'The average, minimum, and maximum completion times are reported for each age group. ',
                                 'tab:time_by_age',
                                 tables_output_folder,
                                 'time_by_age.tex',
                                 std=False)

    # Generate a LaTeX table with the results of the time analysis by artefact
    print('\nGenerating LaTeX table with the results of the time analysis across interaction dimensions')
    time_by_category_latex_table(df, 'ARTEFACT',
                                 'Analysis of activity completion time across interaction dimensions. '
                                 'The table presents a comprehensive overview of the time taken by students '
                                 'to complete all schemas using different interaction dimensions. '
                                 'The average, minimum, and maximum completion times are reported for each '
                                 'interaction dimension.',
                                 'tab:time_by_artefact',
                                 tables_output_folder,
                                 'time_by_artefact.tex',
                                 std=False)


def generate_additional_tables(df, tables_output_folder):
    """
    Generate additional tables
    :param df:                   the DataFrame containing the data
    :param tables_output_folder: the folder where to save the tables
    """

    # Generate a LaTeX tables_output_folder with the results of the time analysis by schema
    print('\nGenerating LaTeX table with the results of the time analysis across schemas')
    time_by_schema_latex_table(df,
                               'Analysis of schema completion time. '
                               'The table presents a detailed breakdown of the average, minimum, '
                               'and maximum completion times for each schema.',
                               'tab:time_by_schema',
                               tables_output_folder, std=False)

    # Generate a LaTeX table with the results of the performance analysis by age category
    print('\nGenerating LaTeX table with the results of the performance analysis across age categories')
    performance_by_category_latex_table(df,
                                        'AGE',
                                        'Analysis of student performance across age categories, detailing the completion and correctness rates. '
                                        'The table presents the number and percentage of students who completed '
                                        'all schemas and correctly solved all schemas. Additionally are reported the median and interquartile range (Q1-Q3) '
                                        'of the number of schemas completed and correctly solved out of the total of 12 schemas, '
                                        'offering a non-parametric summary of central tendency and variability in place of the mean and standard deviation.',
                                        'tab:performance_by_age',
                                        tables_output_folder,
                                        'performance_by_age.tex')

    # Generate a LaTeX table with the results of the algorithm analysis by age category
    print('\nGenerating LaTeX table with the results of the algorithm analysis across age categories')
    algorithms_by_age_latex_table(df,
                                  'Analysis of the algorithms used across age categories. '
                                  'The table presents the number and percentage of different algorithms conceived, '
                                  'along with the distribution of algorithms by their dimensions (0D, 1D, 2D) within '
                                  'each age category.',
                                  'tab:algorithms_by_age',
                                  tables_output_folder)

    # Generate a LaTeX table with the results of the algorithm analysis by schema
    print('\nGenerating LaTeX table with the results of the algorithm analysis across schemas')
    algorithms_by_schema_latex_table(df,
                                     'Analysis of the algorithms used for each schema. '
                                     'The table presents the number and percentage of different algorithms conceived, '
                                     'along with the distribution of algorithms by their dimensions (0D, 1D, 2D) '
                                     'for each schema.',
                                     'tab:algorithms_by_schema',
                                     tables_output_folder)

    # Generate a LaTeX table with the results of the algorithm analysis
    print('\nGenerating LaTeX table with the results of the algorithm analysis')
    algorithms_latex_table(df,
                           'Analysis of the algorithms distribution. '
                           'The table presents the distribution of algorithms conceived by students '
                           'across all schemas. '
                           'The percentages indicate the frequency of each algorithm within each schema.',
                           'tab:algorithms',
                           tables_output_folder)

    # Generate a LaTeX table with the results of the performance analysis by artefact
    print('\nGenerating LaTeX table with the results of the performance analysis across interaction dimensions')
    performance_by_category_latex_table(df,
                                        'ARTEFACT',
                                        'Analysis of student performance across interaction dimensions, detailing the completion and correctness rates. '
                                        'The table presents the number and percentage of students who completed '
                                        'all schemas and correctly solved all schemas. Additionally are reported the median and interquartile range (Q1-Q3) '
                                        'of the number of schemas completed and correctly solved out of the total of 12 schemas, '
                                        'offering a non-parametric summary of central tendency and variability in place of the mean and standard deviation.',
                                        'tab:performance_by_artefact',
                                        tables_output_folder,
                                        'performance_by_artefact.tex')


def analyse_logs(df, output_folders):
    """
    Analyse logs
    :param df:             the DataFrame containing the data
    :param output_folders: the folder where to save the images
    """
    # sns.histplot(data=df, x='RESTARTS')  # Regola il numero di bins se necessario
    # plt.show()

    # compute the mean/median of RESTARTS
    print(df['RESTARTS'].mean())

    # compute the mean/median of restarts by student
    print(df.groupby('STUDENT_ID')['RESTARTS'].mean().mean())

    # Calculate the correlation matrix using columns. How does RESTARTS correlate with the other columns?
    correlation = df[['RESTARTS', 'ALGORITHM_DIMENSION', 'CAT_SCORE', 'SCHEMA_ID', 'AGE_GROUP']].corr()
    print(correlation)

    # # Dispersion graphs
    # sns.scatterplot(data=df, x='RESTARTS', y='ALGORITHM_DIMENSION')
    # plt.show()
    #
    # sns.scatterplot(data=df, x='RESTARTS', y='CAT_SCORE')
    # plt.show()
    #
    # sns.scatterplot(data=df, x='RESTARTS', y='SCHEMA_ID')
    # plt.show()
    #
    # sns.scatterplot(data=df, x='RESTARTS', y='AGE_GROUP')
    # plt.show()

    # Boxplot to visualise the distribution of the data
    # sns.boxplot(x='RESTARTS', y='ALGORITHM_DIMENSION', data=df)
    # plt.show()
    #
    # sns.boxplot(x='RESTARTS', y='CAT_SCORE', data=df)
    # plt.show()

    # Assuming df is your DataFrame and it contains 'SCHEMA' and 'RESTARTS' columns
    # sns.scatterplot(x='SCHEMA_ID', y='RESTARTS', data=df)
    # sns.regplot(x='SCHEMA_ID', y='RESTARTS', data=df, scatter=False, color='red')
    # plt.title('Task Schema vs. Restarts')
    # plt.xlabel('Task Schema')
    # plt.ylabel('Number of Restarts')
    # plt.show()
    # Linear regression model between RESTARTS and SCHEMA_ID
    new_df = df.copy()
    new_df = new_df.dropna(subset=['RESTARTS', 'ALGORITHM_DIMENSION'])

    X = sm.add_constant(new_df['SCHEMA_ID'])
    schema_model = sm.OLS(new_df['RESTARTS'], X).fit()
    print('Linear regression model between RESTARTS and SCHEMA_ID', schema_model.summary())

    # Extracting coefficients for plotting
    # coef = schema_model.params
    # conf = schema_model.conf_int()
    # conf['coef'] = coef

    # # Coefficient plot with confidence intervals
    # plt.figure(figsize=(8, 4))
    # plt.errorbar(x=conf.index, y='coef', yerr=[conf[0], conf[1]], data=conf, fmt='o', capsize=5)
    # plt.title('Coefficient Plot of Schema Model')
    # plt.xlabel('Variable')
    # plt.ylabel('Coefficient Value')
    # plt.axhline(y=0, linestyle='--', color='grey')
    # plt.show()
    #
    # # Predicted vs. Actual Plot
    # new_df['Predicted_RESTARTS-schema'] = schema_model.predict(X)
    #
    # plt.figure(figsize=(8, 4))
    # plt.scatter(new_df['RESTARTS'], new_df['Predicted_RESTARTS-schema'], alpha=0.3)
    # plt.plot([new_df['RESTARTS'].min(), new_df['RESTARTS'].max()],
    #          [new_df['Predicted_RESTARTS-schema'].min(), new_df['Predicted_RESTARTS-schema'].max()],
    #          color='red', lw=2)
    # plt.title('Predicted vs. Actual Restarts')
    # plt.xlabel('Actual Restarts')
    # plt.ylabel('Predicted Restarts')
    # plt.show()
    #
    # # Residual Plot
    # residuals = schema_model.resid
    #
    # plt.figure(figsize=(8, 4))
    # plt.scatter(new_df['Predicted_RESTARTS-schema'], residuals, alpha=0.3)
    # plt.axhline(y=0, linestyle='--', color='grey')
    # plt.title('Residual Plot')
    # plt.xlabel('Predicted Restarts')
    # plt.ylabel('Residuals')
    # plt.show()

    generate_restart_distribution_plot(new_df, 'SCHEMA_ID', 'Schema', output_folders,
                                       order=list(range(1, 13)))

    # Linear regression model between RESTARTS and AGE
    X = sm.add_constant(new_df['AGE'])
    age_model = sm.OLS(new_df['RESTARTS'], X).fit()
    print('Linear regression model between RESTARTS and AGE', age_model.summary())

    generate_restart_distribution_plot(new_df, 'AGE_CATEGORY', 'Age category', output_folders,
                                       order=['From 3 to 6 years old', 'From 7 to 9 years old',
                                              'From 10 to 13 years old', 'From 14 to 16 years old'],
                                       x_ticks=['3-6', '7-9', '10-13', '14-16'])

    # Linear regression model between RESTARTS and gender
    X = sm.add_constant(new_df['GENDER_IDX'])
    gender_model = sm.OLS(new_df['RESTARTS'], X).fit()
    print('Linear regression model between RESTARTS and gender', gender_model.summary())

    # mean_restarts_per_gender.plot(kind='bar')
    # plt.ylabel('Media RESTARTS')
    # plt.title('Media RESTARTS per GENDER')
    # plt.show()

    # sns.boxplot(x='RESTARTS', y='GENDER_IDX', data=df)
    # plt.show()

    # Tendence analysis
    # Create groups of RESTARTS
    # bins = [0, 1, 5, 10, 15]
    # labels = ['0', '1-4', '5-9', '10-15']
    # df['RESTARTS_GROUP'] = pd.cut(df['RESTARTS'], bins=bins, labels=labels, right=False)
    #
    # # Compute the mean of ALGORITHM_DIMENSION and CAT_SCORE for each group of RESTARTS
    # grouped = df.groupby('RESTARTS_GROUP', observed=True)[['ALGORITHM_DIMENSION', 'CAT_SCORE']].mean()
    #
    # # Grafico a barre
    # grouped.plot(kind='bar')
    # plt.ylabel('Media')
    # plt.title('Media di ALGORITHM_DIMENSION e CAT_SCORE per Gruppo di RESTARTS')
    # plt.show()

    # Linear regression model between RESTARTS and ARTEFACTS
    X = sm.add_constant(new_df['ARTEFACT_DIMENSION'])
    artefact_model = sm.OLS(new_df['RESTARTS'], X).fit()
    print('Linear regression model between RESTARTS and ARTEFACT_DIMENSION', artefact_model.summary())

    X = sm.add_constant(new_df['PREDOMINANT_ARTEFACT_DIMENSION'])
    predominant_artefact_model = sm.OLS(new_df['RESTARTS'], X).fit()
    print('Linear regression model between RESTARTS and PREDOMINANT_ARTEFACT_DIMENSION', predominant_artefact_model.summary())

    generate_restart_distribution_plot(new_df, 'ARTEFACT_TYPE', 'Lowest interaction dimension', output_folders,
                                       order=['GF', 'G', 'PF', 'P'],
                                       x_ticks=['GF', 'G', 'PF', 'P'])

    generate_restart_distribution_plot(new_df, 'PREDOMINANT_ARTEFACT_TYPE', 'Prevalent interaction dimension', output_folders,
                                       order=['GF', 'G', 'PF', 'P'],
                                       x_ticks=['GF', 'G', 'PF', 'P'])

    # Linear regression model between RESTARTS and ALGORITHM_DIMENSION
    X = sm.add_constant(new_df['RESTARTS'])
    model = sm.OLS(new_df['ALGORITHM_DIMENSION'], X).fit()
    print('Linear regression model between RESTARTS and ALGORITHM_DIMENSION', model.summary())

    # Linear regression model between RESTARTS and CAT_SCORE
    model = sm.OLS(new_df['CAT_SCORE'], X).fit()
    print('Linear regression model between RESTARTS and CAT_SCORE', model.summary())

    # Adding polynomial terms
    new_df['RESTARTS_Squared'] = new_df['RESTARTS'] ** 2
    new_df['RESTARTS_Cubed'] = new_df['RESTARTS'] ** 3

    # Building the model
    X = new_df[['RESTARTS', 'RESTARTS_Squared', 'RESTARTS_Cubed']]
    y = new_df['CAT_SCORE']

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())

    y = new_df['ALGORITHM_DIMENSION']
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Defining a threshold (this should be based on your data and hypothesis)
    threshold = 10  # Example threshold

    # Segmenting the data
    df_below_threshold = new_df[new_df['RESTARTS'] <= threshold]
    df_above_threshold = new_df[new_df['RESTARTS'] > threshold]

    # Fitting a linear model to data below the threshold
    X_below = df_below_threshold[['RESTARTS']]
    y_below = df_below_threshold['CAT_SCORE']
    model_below = LinearRegression().fit(X_below, y_below)

    # Fitting a linear model to data above the threshold
    X_above = df_above_threshold[['RESTARTS']]
    y_above = df_above_threshold['CAT_SCORE']
    model_above = LinearRegression().fit(X_above, y_above)

    # You can then analyze and compare the coefficients, intercepts, and performance of these models
    print("Model below threshold:", model_below.coef_, model_below.intercept_)
    print("Model above threshold:", model_above.coef_, model_above.intercept_)

    # Segmenting the data
    df_below_threshold = new_df[new_df['RESTARTS'] <= threshold]
    df_above_threshold = new_df[new_df['RESTARTS'] > threshold]

    # Fitting a linear model to data below the threshold
    X_below = df_below_threshold[['RESTARTS']]
    y_below = df_below_threshold['ALGORITHM_DIMENSION']
    model_below = LinearRegression().fit(X_below, y_below)

    # Fitting a linear model to data above the threshold
    X_above = df_above_threshold[['RESTARTS']]
    y_above = df_above_threshold['ALGORITHM_DIMENSION']
    model_above = LinearRegression().fit(X_above, y_above)

    # You can then analyze and compare the coefficients, intercepts, and performance of these models
    print("Model below threshold:", model_below.coef_, model_below.intercept_)
    print("Model above threshold:", model_above.coef_, model_above.intercept_)


def main(input_folder, output_folders, tables_output_folder, schemas_folder, study):
    """
    Main function
    :param input_folder:         the folder containing the tables
    :param output_folders:       the folder where to save the images
    :param tables_output_folder: the folder where to save the tables
    :param schemas_folder:       the folder containing the schemas
    :param study:                the type of experimental study (pilot or main)
    """
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.width', None)  # No max width

    if not os.path.exists(tables_output_folder):
        os.makedirs(tables_output_folder)

    # Import the tables
    dataframes = import_tables(input_folder)

    for d in dataframes:
        print_df_info(dataframes[d])
        print(dataframes[d].head())
        print()

    df = dataframes['DF']

    # Calculate the number of unique STUDENT_ID for each session (SCHEMA_ID)
    num_students_per_session = df.groupby('SESSION_ID')['STUDENT_ID'].nunique().reset_index()
    print(num_students_per_session)

    # Print info, describe and missing values for each DataFrame in used_dfs
    print_df_info(df)

    df['ARTEFACT_TYPE'] = df['ARTEFACT_TYPE'].map({
        '0 (GF)': 'GF',
        '1 (G)': 'G',
        '2 (PF)': 'PF',
        '3 (P)': 'P'
    })

    df['AGE_CATEGORY'] = df['AGE_CATEGORY'].map({
        '3-6': 'From 3 to 6 years old',
        '7-9': 'From 7 to 9 years old',
        '10-13': 'From 10 to 13 years old',
        '14-16': 'From 14 to 16 years old'
    })

    if study == 'pilot':
        # Change the SESSION_ID from 4 and 5 to 3
        df['SESSION_ID'] = df['SESSION_ID'].replace(4, 3)
        df['SESSION_ID'] = df['SESSION_ID'].replace(5, 3)

        # Change the Harmos grade from 0 and 1 and 2 to 0, 1, 2
        df['HARMOS_GRADE'] = df['HARMOS_GRADE'].replace(0, '0, 1, 2')
        df['HARMOS_GRADE'] = df['HARMOS_GRADE'].replace(1, '0, 1, 2')
        df['HARMOS_GRADE'] = df['HARMOS_GRADE'].replace(2, '0, 1, 2')

        # change the school id from 3 to X and from 4 to Y
        df['SCHOOL_ID'] = df['SCHOOL_ID'].replace(3, 'X')
        df['SCHOOL_ID'] = df['SCHOOL_ID'].replace(4, 'Y')

    # For each student, compute the frequency of COMPLETE == True
    frequency_of_complete = df.groupby('STUDENT_ID')['COMPLETE'].apply(lambda x: x[x == True].count()).reset_index()
    # Remove the STUDENT_ID column
    frequency_of_complete = frequency_of_complete.iloc[:, 1]

    # Compute the average performance of the students
    avg_performance = df.groupby('STUDENT_ID')['CAT_SCORE'].mean().reset_index()
    avg_performance = avg_performance.iloc[:, 1]

    # Add the column for the constant term
    frequency_of_complete = sm.add_constant(frequency_of_complete)

    model = sm.OLS(avg_performance, frequency_of_complete)
    results = model.fit()
    print(results.summary())

    # Compute the percentage of complete, so the number of complete over the total number of rows of the DataFrame
    engagement = df['COMPLETE'].sum() / df.shape[0]

    # Compute the percentage of correct over the completed
    correct = df[df['COMPLETE'] == True]['CORRECT'].sum() / df['COMPLETE'].sum()

    # Compute the average session duration (sum of time for student, averaged over the number of students)
    avg_session_duration = df.groupby('STUDENT_ID')['LOG_TIME'].sum().mean()
    avg_session_duration = format_time(avg_session_duration)

    # Analyse logs: TODO: tested only for the main study, test on the pilot study
    if study == 'main':
        logs_df = plot_interaction_dimension(dataframes['LOGS'], output_folders)

        # Add the column PREDOMINANT_ARTEFACT_TYPE to df from log_df merging on RESULT_ID
        df = pd.merge(df, logs_df[['RESULT_ID', 'PREDOMINANT_ARTEFACT_TYPE', 'PREDOMINANT_ARTEFACT_DIMENSION']],
                      on='RESULT_ID', how='left')

        # For each RESULT_ID count the occurrences of commandsReset in logs_df.DESCRIPTIONS
        # if no occurrence set 0, otherwise set the number of occurrences
        commands_reset = logs_df.groupby('RESULT_ID')['DESCRIPTIONS'].apply(
            lambda x: x.str.count('commandsReset')).reset_index()

        # Append to the df the column COMMANDS_RESET and use the values in the DESCRIPTIONS column from commands_reset
        df['RESTARTS'] = commands_reset['DESCRIPTIONS']

        analyse_logs(df, output_folders)

    # Generate plots
    print('\nGenerating plots')
    generate_plots(df, schemas_folder, output_folders)

    # Generate paper tables
    print('\nGenerating paper tables')
    generate_paper_tables(df, tables_output_folder)

    # Generate additional tables
    print('\nGenerating additional tables')
    generate_additional_tables(df, tables_output_folder)

    generate_time_plot(df, output_folders)

    # Export dataframe for linear mixed-effects analysis
    # TODO: ARMOS_GRADE - BIRTH YEAR (...)
    if study == 'main':
        df = df[['STUDENT_ID', 'SCHEMA_ID', 'LOG_TIME', 'ARTEFACT_DIMENSION', 'ALGORITHM_DIMENSION', 'AGE_CATEGORY',
                 'AGE', 'GENDER', 'SESSION_ID', 'HARMOS_GRADE', 'SCHOOL_ID', 'CANTON_NAME', 'CAT_SCORE']]
    else:
        df = df[['STUDENT_ID', 'SCHEMA_ID', 'ARTEFACT_DIMENSION', 'ALGORITHM_DIMENSION', 'AGE_CATEGORY',
                 'GENDER', 'SESSION_ID', 'HARMOS_GRADE', 'SCHOOL_ID', 'CANTON_NAME', 'CAT_SCORE']]

    # Add column DOMAIN and assign to all rows the value Virtual
    df = df.copy()
    df['DOMAIN'] = 'Virtual'

    # Make all columns categorical except CAT_SCORE, LOG_TIME
    df = df.astype({col: 'category' for col in df.columns if col not in ['CAT_SCORE', 'LOG_TIME']})
    df = df.dropna()

    df.to_csv(os.path.join(input_folder, 'lme_dataset_{}.csv'.format(study)), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute data analysis.')
    parser.add_argument('--input_folder', required=True,
                        help='Path to the folder containing the input tables.')
    parser.add_argument('--output_folders', required=True, nargs='+',
                        help='Path to the folder where to save the images.')
    parser.add_argument('--tables_output_folder', required=True,
                        help='Path to the folder where to save the LaTeX tables.')
    parser.add_argument('--schemas_folder', required=True,
                        help='Path to the folder containing the schemas images.')
    parser.add_argument('--study', required=True,
                        help='The type of experimental study (pilot or main).')

    args = parser.parse_args()

    main(args.input_folder, args.output_folders, args.tables_output_folder, args.schemas_folder, args.study)
