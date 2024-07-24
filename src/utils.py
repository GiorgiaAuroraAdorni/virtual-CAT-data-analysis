"""
@file-name:     utils.py
@date-creation: 07.07.2023
@author-name:   Giorgia Adorni
"""
import datetime
import glob
import os
import re

import numpy as np
import pandas as pd


def import_tables(folder_path):
    """
    Import all the CSV files in the folder as DataFrames
    :param folder_path: the path of the folder containing the CSV files
    :return:            a dictionary of DataFrames
    """
    # Get the paths of all CSV files in the folder
    csv_files = glob.glob(folder_path + '*.csv')

    # Create a dictionary to store the DataFrames
    dataframes = {}

    # Iterate over the CSV files and load them into DataFrames
    for file in csv_files:
        # Get the file name without extension
        file_name = file.split('/')[-1].split('.')[0]

        # Read the CSV file directly into the dataframes dictionary
        dataframes[file_name] = pd.read_csv(file)

    # Access the DataFrames
    print(f'Loaded {len(dataframes)} files.')
    print('Dataframes created:')
    # for file_name in dataframes.keys():
    #     print(file_name)

    return dataframes


def import_png_images(folder_path):
    """
    Import all the images in the folder as DataFrames
    :param folder_path: the path of the folder containing the images
    :return:            two lists of images paths and names
    """
    paths = []
    names = []

    for file in os.listdir(folder_path):
        if file.endswith('.png'):
            file_path = os.path.join(folder_path, file)
            paths.append(file_path)
            names.append(file.split('.')[0])

    return paths, names


def export_tables(dataframes, folder_path):
    """
    Export the DataFrames as CSV files
    :param dataframes:  a dictionary of DataFrames
    :param folder_path: the path of the folder where to save the CSV files
    :return:            None
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Export each dataframe as a CSV file
    for df_name, df in dataframes.items():
        output_file = os.path.join(folder_path, f'{df_name}.csv')
        df.to_csv(output_file, index=False)


def anonymise_dataframes(dataframes, folder_path):
    """
    Anonymise the DataFrames:
    - The ALGORITHMS, STUDENTS_SESSIONS, DF, CANTONS, LOGS, RESULTS, STUDENTS dataframes remain as they are
    - Remove the SCHOOL_NAME column from the SCHOOLS dataframe.
    - Remove the SECTION and NOTES columns SESSIONS dataframe.
    :param dataframes: dictionary of DataFrames
    :param folder_path: path to the output folder
    :return: dictionary of DataFrames
    """
    new_dataframes = {}
    new_dataframes['ALGORITHMS'] = dataframes['ALGORITHMS']
    new_dataframes['SCHOOLS'] = dataframes['SCHOOLS'].drop(columns=['SCHOOL_NAME'])
    new_dataframes['STUDENTS_SESSIONS'] = dataframes['STUDENTS_SESSIONS']
    new_dataframes['DF'] = dataframes['DF']
    new_dataframes['CANTONS'] = dataframes['CANTONS']
    new_dataframes['LOGS'] = dataframes['LOGS']
    new_dataframes['RESULTS'] = dataframes['RESULTS']
    new_dataframes['SESSIONS'] = dataframes['SESSIONS'].drop(columns=['SECTION', 'NOTES'])

    # If there is the SURVEY dataframe, add it to the new_dataframes
    if 'SURVEY' in dataframes.keys():
        new_dataframes['SURVEY'] = dataframes['SURVEY']
    export_tables(new_dataframes, folder_path)

    return new_dataframes


def calculate_age(birth_date, current_date):
    """
    Calculate the age of the user
    :param birth_date:   the birthdate
    :param current_date: the current date
    :return:             the age of the user
    """
    birth_datetime = datetime.datetime.strptime(birth_date, '%Y-%m-%d')
    current_datetime = datetime.datetime.strptime(current_date, '%Y-%m-%d')
    age = current_datetime.year - birth_datetime.year
    if current_datetime.month < birth_datetime.month or (
            current_datetime.month == birth_datetime.month and current_datetime.day < birth_datetime.day):
        age -= 1
    return age


def replace_old_column_with_new_one(df, old_column, new_column):
    """
    Replace the old column with the new one in the DataFrame
    :param df:         the DataFrame
    :param old_column: the name of the old column
    :param new_column: the name of the new column
    :return:           the DataFrame with the old column replaced with the new one
    """
    # Get the index of the old column in the DataFrame
    column_index = df.columns.get_loc(old_column)

    # Move the new column to the right of the old column
    df.insert(column_index + 1, new_column, df.pop(new_column))

    # Remove the old column from the DataFrame
    df.pop(old_column)

    # Rename the new column to the old column name
    df.rename(columns={new_column: old_column}, inplace=True)

    return df


def remove_extreme_elements(descriptions, interfaces, feedbacks, commands, remove_last=True):
    """
    Remove the last element from the input lists if the last description is 'changeVisibility'
    :param descriptions: the list of descriptions
    :param interfaces:   the list of interfaces
    :param feedbacks:    the list of feedbacks
    :param commands:     the list of commands
    :param remove_last:  if True, remove the last element; if False, remove the first element
    :return:             None
    """
    if not descriptions:
        return

    if remove_last:
        idx = -1
    else:
        idx = 0

    if descriptions[idx] == 'changeVisibility':
        descriptions.pop(idx)
        interfaces.pop(idx)
        feedbacks.pop(idx)
        commands.pop(idx)
        remove_extreme_elements(descriptions, interfaces, feedbacks, commands, remove_last)


def fix_timestamp_format(json_str):
    """
    Fix the incorrect timestamp format in the JSON string
    :param json_str:  the JSON string
    :return:          the JSON string with the correct timestamp format
    """
    # Use regular expression to find and fix incorrect timestamp format
    fixed_json_str = re.sub(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})(?=:)',
                            '"\g<1>"', json_str)

    return fixed_json_str


def count_changes_and_indices(data):
    """
    Return the number of changes in the input data and the list of unique values
    :param data: the input data (list)
    :return:     the number of changes and the indices of the changes
    """
    prev_value = data[0]
    change_count = 0
    unique_values = [prev_value]

    for value in data[1:]:
        if value != prev_value:
            change_count += 1
            unique_values.append(value)
        prev_value = value

    return change_count, unique_values


def remove_occurrences_on_condition(commands, descriptions, feedbacks, interfaces,
                                    list_to_be_used=None, condition='changeMode'):
    """
    Remove the occurrences of changeMode from the input lists
    :param commands:        the list of commands
    :param descriptions:    the list of descriptions
    :param feedbacks:       the list of feedbacks
    :param interfaces:      the list of interfaces
    :param list_to_be_used: the list to be used for removing the occurrences, default is descriptions
    :param condition:       the condition to be used for removing the occurrences, default is 'changeMode'
    :return:                None
    """
    if list_to_be_used is None:
        list_to_be_used = descriptions

    # Identify indices to remove
    indices_to_remove = []
    for idx, el in enumerate(list_to_be_used):
        if el in {condition} and descriptions[idx] != 'commandsReset':
            # The following is not necessary since in extract_algorithm_from_log in
            # logs_data_preparation.py when the description is buttonSelect everything is ignored
            # if condition == 'buttonDismiss':
            # if descriptions[idx - 1] == 'buttonSelect':
            #     print(descriptions[idx - 1], '\t', descriptions[idx])
            #     print(commands[idx - 1], '\t', commands[idx])
            #     print('\n')
            #     # append also the previous index
            #     indices_to_remove.append(idx - 1)

            # if condition == 'dismissCommand':
            #     print(descriptions[idx - 1], '\t', descriptions[idx])
            #     print(commands[idx - 1], '\t', commands[idx])
            #     print('\n')

            indices_to_remove.append(idx)

    # Remove elements from lists
    for idx in reversed(indices_to_remove):
        descriptions.pop(idx)
        interfaces.pop(idx)
        feedbacks.pop(idx)
        commands.pop(idx)


def merge_dataframes(results_df, algorithms_df, students_sessions_df):
    """
    Merge the RESULTS, ALGORITHMS and STUDENTS_SESSIONS DataFrames
    :param results_df:           the RESULTS DataFrame
    :param algorithms_df:        the ALGORITHMS DataFrame
    :param students_sessions_df: the STUDENTS_SESSIONS DataFrame
    :return:                     the merged DataFrame
    """
    # Merge all the DataFrames
    merged_df = pd.merge(results_df, algorithms_df, on='ALGORITHM_ID', how='left')

    # Remove the SCHEMA_ID_y column
    merged_df = merged_df.drop(columns=['SCHEMA_ID_y'])

    # Rename SCHEMA_ID_x to SCHEMA_ID
    merged_df = merged_df.rename(columns={'SCHEMA_ID_x': 'SCHEMA_ID'})

    # Remove the ALGORITHM_y column
    merged_df = merged_df.drop(columns=['ALGORITHM_y'])

    # Rename ALGORITHM_x to ALGORITHM
    merged_df = merged_df.rename(columns={'ALGORITHM_x': 'ALGORITHM'})

    common_columns = merged_df.columns.intersection(students_sessions_df.columns).tolist()
    merged_df = pd.merge(merged_df, students_sessions_df, on=common_columns, how='left')

    return merged_df


def compute_cat_score(dataframe):
    """
    Compute the CAT score for each row in the DataFrame
    :param dataframe: the DataFrame
    :return:          the DataFrame with the CAT score computed
    """
    dataframe['CAT_SCORE'] = dataframe['ALGORITHM_DIMENSION'] + dataframe['ARTEFACT_DIMENSION']

    return dataframe


def count_commands_dimension(algorithm):
    """
    Compute the dimension of the algorithm based on the commands it contains
    :param algorithm: the algorithm
    :return:          the dimension of the algorithm
    """
    # If the algorithm is nan, return nan
    if algorithm is np.nan:
        return np.nan

    regex_dot = r'paint\((yellow|blue|red|green)\)'
    regex_monochromatic_pattern = r'paint\(\{(?:yellow|blue|red|green)\},(?:([1-6:],[\w\s-]+)|\{.*\})\)'
    regex_polychromatic_pattern = r'paint\(\{(?:yellow|blue|red|green)(?:,(?:yellow|blue|red|green))+\},(?:([1-6:],[\w\s-]+)|\{.*\})\)'
    regex_fillempty = r'fill_empty\((yellow|blue|red|green)\)'
    regex_copy = r'copy\(\{.*\},\{.*\}\)'
    regex_mirror = r'mirror\((\{.*\})?,?(horizontal|vertical)\)'
    regex_wrong_paint = r'paint\(\{\},[1-6:],[\w\s-]+\)'

    # Count the number of 0D commands
    d0_commands = re.findall(regex_dot, algorithm)

    # Count the number of 1D commands
    d1_commands = re.findall(regex_monochromatic_pattern, algorithm) + re.findall(regex_fillempty, algorithm) + re.findall(regex_wrong_paint, algorithm)

    # Count the number of 2D commands
    d2_commands = re.findall(regex_polychromatic_pattern, algorithm) + re.findall(regex_copy, algorithm) + re.findall(regex_mirror, algorithm)

    # Count the total number of commands
    total_commands = len(d0_commands) + len(d1_commands) + len(d2_commands)

    if total_commands == 0:
        return 0

    score = (3 * len(d2_commands)/total_commands) + (2 * len(d1_commands)/total_commands) + (1 * len(d0_commands)/total_commands)

    # Get the maximum dimension of the commands
    # (if len(d2_commands) > 0, then max_dimension = 2,
    # else if len(d1_commands) > 0, then max_dimension = 1,
    # else max_dimension = 0)
    max_dimension = 3 if len(d2_commands) > 0 else 2 if len(d1_commands) > 0 else 1

    final_score = score + max_dimension/total_commands + 1/total_commands

    return final_score

def compute_new_score(dataframe):
    """
    Compute the CAT score for each row in the DataFrame
    :param dataframe: the DataFrame
    :return:          the DataFrame with the CAT score computed
    """
    # Get the new score
    dataframe['WEIGHTED_ALGORITHM_DIMENSION'] = dataframe['ALGORITHM'].apply(count_commands_dimension)

    # print(dataframe['WEIGHTED_ALGORITHM_DIMENSION'].min())
    # print(dataframe['WEIGHTED_ALGORITHM_DIMENSION'].max())

    # Rescale the WEIGHTED_ALGORITHM_DIMENSION from 0-6 to 0-2
    dataframe['WEIGHTED_ALGORITHM_DIMENSION'] = dataframe['WEIGHTED_ALGORITHM_DIMENSION'].apply(lambda x: x * 2 / 6)

    # print(dataframe['WEIGHTED_ALGORITHM_DIMENSION'].min())
    # print(dataframe['WEIGHTED_ALGORITHM_DIMENSION'].max())
    #
    # # count the occurrences of the max value in the column
    # max_value_count = dataframe['WEIGHTED_ALGORITHM_DIMENSION'].value_counts().max()
    # print('Max value count: ', max_value_count)

    # compute the new weighted CAT score
    dataframe['WEIGHTED_CAT_SCORE'] = dataframe['WEIGHTED_ALGORITHM_DIMENSION'] + dataframe['ARTEFACT_DIMENSION']

    return dataframe


def fill_dimensions_nan(dataframe):
    """
    Fill the NaN values in the columns ALGORITHM_DIMENSION and ARTEFACT_DIMENSION
    :param dataframe: the DataFrame
    :return:          the DataFrame with the NaN values filledwith -1
    """
    # Replace NaN values with -1 in the columns ALGORITHM_DIMENSION and ARTEFACT_DIMENSION
    dataframe['ALGORITHM_DIMENSION'].fillna(-1, inplace=True)
    dataframe['ARTEFACT_DIMENSION'].fillna(-1, inplace=True)

    return dataframe


def assign_schema_factor(dataframe):
    """
    Assign a numerical value to the column FACTOR
    :param dataframe: the DataFrame containing the data
    :return:          the DataFrame containing the data with the column FACTOR
    """
    factor_mapping = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 3, 8: 3, 9: 3, 10: 2, 11: 2, 12: np.nan}
    dataframe['FACTOR'] = dataframe['SCHEMA_ID'].map(factor_mapping)

    return dataframe


def assign_school_level(dataframe):
    """
    Assign a numerical value to the column SCHOOL_LEVEL
    :param dataframe: the DataFrame containing the data
    :return:          the DataFrame containing the data with the column SCHOOL_LEVEL
    """
    school_mapping = {'Preschool': 0, 'Primary school': 1, 'Secondary school': 2, 'Low secondary school': 2}

    dataframe['SCHOOL_LEVEL'] = dataframe['SCHOOL_TYPE'].map(school_mapping)

    return dataframe


def assign_gender_idx(dataframe):
    """
    Assign a numerical value to the column GENDER_IDX
    :param dataframe: the DataFrame containing the data
    :return:          the DataFrame containing the data with the column GENDER_IDX
    """
    gender_mapping = {'Male': 0, 'Female': 1}

    dataframe['GENDER_IDX'] = dataframe['GENDER'].map(gender_mapping)

    return dataframe


def assign_schema_name(dataframe):
    """
    Assign a numerical value to the column SCHEMA_NAME
    :param dataframe: the DataFrame containing the data
    :return:          the DataFrame containing the data with the column SCHEMA_NAME
    """
    dataframe['SCHEMA_NAME'] = 'S' + dataframe['SCHEMA_ID'].astype(str)

    return dataframe


def print_df_info(df):
    """
    Print the info, describe and missing values of the DataFrame
    :param df: the DataFrame to print info, describe and missing values
    """
    print('Info\n', df.info())
    print('Describe\n', df.describe())
    print('Missing values\n', df.isna().sum())


def get_count_and_percentage(df, column_name):
    """
    Get the string containing the count and percentage of the column
    :param df:          the DataFrame
    :param column_name: the column name
    :return:            the string containing the count and percentage of the column
    """
    # Count the number of STUDENT_ID where COMPLETE is TRUE
    complete_count = df[df[column_name] == True]['STUDENT_ID'].nunique()

    if complete_count == 0:
        return f'{complete_count} (0\%)'

    # Calculate the percentage based on the total count
    complete_percentage = round((complete_count / df['STUDENT_ID'].nunique()) * 100)

    return f'{complete_count} ({complete_percentage:}\%)'


def save_and_print_table(latex_code, caption, label, out_folder, filename):
    """
    Save LaTeX code to a file and print it
    :param latex_code: The LaTeX code to save
    :param caption:    The caption of the table
    :param label:      The label of the table
    :param out_folder: The path of the folder where to save the file
    :param filename:   The name of the file
    :return:           None
    """
    # Add begin table, caption, label and end table
    latex_code = latex_code.replace(r'\begin{tabular}', r'\begin{table*}[h]\footnotesize\centering'
                                                        r'\caption{' + caption + r'}\label{' + label + r'}'
                                                        r'\begin{tabular}')
    latex_code = latex_code.replace(r'\end{tabular}', r'\end{tabular}\end{table*}')

    # Save LaTeX code to a file
    with open(out_folder + filename, 'w') as f:
        f.write(latex_code)

    # Print the LaTeX code
    print(latex_code)


def format_time(time):
    """
    Format the time in minutes and seconds
    :param time: the time in seconds
    :return:     the formatted time in minutes and seconds
    """
    return f'{int(time // 60):02d}m {int(time % 60):02d}s'


def format_small_time(time):
    """
    Format the time in minutes and seconds
    :param time: the time in seconds
    :return:     the formatted time in minutes and seconds
    """
    return f'{int(time // 60)} min {int(time % 60):02d} sec'


def compare_and_print_difference(logs_df, column_name, retrieved_column_name, debug=False):
    """
    Compare the values of the column with the retrieved column and print the differences
    :param logs_df:               the LOGS DataFrame
    :param column_name:           the name of the column to compare
    :param retrieved_column_name: the name of the retrieved column
    :param debug:                 if True, print the feedback, interface, artefact type and retrieved artefact type
    :return:                      the list of different IDs
    """
    # Filter out rows with nan values in retrieved_column_name
    filtered_logs_df = logs_df[~logs_df[retrieved_column_name].isna()]

    # Get the rows where the column and the retrieved column are different
    different_values = filtered_logs_df[filtered_logs_df[column_name] != filtered_logs_df[retrieved_column_name]]

    different_ids = 0

    if len(different_values) > 0:
        message = column_name + ' and ' + retrieved_column_name + ' are different.'
        different_ids = different_values['RESULT_ID'].tolist()

        print('\n' + message)
        print('Number of different {}: {}'.format(column_name.lower(), len(different_values)))
        print('Different IDs: ', different_ids)

        # If column_name is ARTEFACT_DIMENSION, compare the ARTEFACT_DIMENSION and RETRIEVED_ARTEFACT_DIMENSION,
        # and count how many times the second is higher and lower than the first
        if column_name == 'ARTEFACT_DIMENSION':
            # Count how many times the retrieved artefact dimension is higher and lower than the artefact dimension
            higher_count = (different_values['RETRIEVED_ARTEFACT_DIMENSION'] >
                            different_values['ARTEFACT_DIMENSION']).sum()
            lower_count = (different_values['RETRIEVED_ARTEFACT_DIMENSION'] <
                           different_values['ARTEFACT_DIMENSION']).sum()

            print('Higher count: ', higher_count)
            print('Lower count: ', lower_count)
            print('\n')

        if debug:
            # For each element with different values, print feedback, interface,
            # artefact type and retrieve artefact type
            for index, row in different_values.iterrows():
                print('\nFeedback: ', row['FEEDBACKS'])
                print('Interface: ', row['INTERFACES'])
                print('Artefact type: ', row['ARTEFACT_TYPE'])
                print('Retrieved artefact type: ', row['RETRIEVED_ARTEFACT_TYPE'])

    else:
        print('\n' + column_name + ' and ' + retrieved_column_name + ' are the same.')

    return different_ids


def map_missing_algorithm(row):
    """
    Map the missing algorithm to the correct algorithm
    :param row: the row of the DataFrame
    :return:    the correct algorithm
    """
    if row['ALGORITHM'] == '':
        return True
    elif pd.isna(row['ALGORITHM']):
        return pd.NA
    else:
        return False


def is_ordered_subset(sublist, full_list):
    """
    Check if the sublist is an ordered subset of the full_list
    :param sublist:   the sublist
    :param full_list: the full list
    :return:          True if the sublist is an ordered subset of the full_list, False otherwise, and the index of the first element of the sublist in the full_list
    """
    i, j = 0, 0
    index_first_match = None
    while i < len(sublist) and j < len(full_list):
        if sublist[i] == full_list[j]:
            i += 1
            if index_first_match is None:
                index_first_match = j
        j += 1

    boolean_match = i == len(sublist)
    return boolean_match, index_first_match


def filter_invalid_copy_commands(retrieved_algorithm, commands_retrieved_algorithm):
    """
    Filter the invalid copy commands from the retrieved algorithm
    :param retrieved_algorithm:          the retrieved algorithm
    :param commands_retrieved_algorithm: the commands of the retrieved algorithm
    :return:                             the filtered retrieved algorithm and the filtered commands of the retrieved algorithm
    """

    regex_invalid_copy = r'copy\(\{\},\{\}\)'
    filtered_retrieved_algorithm = re.sub(regex_invalid_copy, '', retrieved_algorithm)

    if retrieved_algorithm != filtered_retrieved_algorithm:
        filtered_retrieved_algorithm = remove_initial_comma_in_algorithm(filtered_retrieved_algorithm)
        filtered_retrieved_algorithm = remove_multiple_commas_in_algorithm(filtered_retrieved_algorithm)
        filtered_retrieved_algorithm = remove_final_comma_in_algorithm(filtered_retrieved_algorithm)

        filtered_commands_retrieved_algorithm = get_commands_list(filtered_retrieved_algorithm)

        return filtered_retrieved_algorithm, filtered_commands_retrieved_algorithm
    else:
        return retrieved_algorithm, commands_retrieved_algorithm


def fix_algorithms(log_series):
    """
    Fix the algorithms in the log_series
    :param log_series: the log_series
    :return:           the log_series with the fixed algorithms
    """
    # Get all the retrieved algorithms in different_algorithms_df that do not have the go(..) command
    # Check if the retrieved algorithm does not have the go(..) command
    original_retrieved_algorithm = log_series.RETRIEVED_ALGORITHM
    original_algorithm = log_series.ALGORITHM

    # Define all the regular expressions to be used
    regex_go = r'go\(.*\)'
    regex_dot = r'paint\((yellow|blue|red|green)\)'
    regex_pattern = r'paint\(\{(?:yellow|blue|red|green)(?:,(?:yellow|blue|red|green))*\},[1-6:],[\w\s-]+\)'
    regex_fillempty = r'fill_empty\((yellow|blue|red|green)\)'

    # Get the commands list from the algorithms
    original_commands_algorithm = get_commands_list(original_algorithm)
    original_commands_retrieved_algorithm = get_commands_list(original_retrieved_algorithm)

    # Check the type of commands present in the retrieved algorithm
    found_fillempty = re.search(regex_fillempty, original_retrieved_algorithm)

    # Copy the original values
    commands_retrieved_algorithm = original_commands_retrieved_algorithm.copy()
    commands_algorithm = original_commands_algorithm.copy()
    algorithm = ','.join(commands_algorithm)
    retrieved_algorithm = ','.join(commands_retrieved_algorithm)

    # Check for non-existing go coordinates
    algorithm, commands_algorithm = filter_invalid_go_commands(algorithm, commands_algorithm, log_series.LOG_ID)
    retrieved_algorithm, commands_retrieved_algorithm = filter_invalid_go_commands(retrieved_algorithm,
                                                                                   commands_retrieved_algorithm,
                                                                                   log_series.LOG_ID)
    # Remove consecutive go commands and join the commands with a comma to form the algorithm
    algorithm, commands_algorithm = remove_consecutive_go(commands_algorithm)
    retrieved_algorithm, commands_retrieved_algorithm = remove_consecutive_go(commands_retrieved_algorithm,
                                                                              retrieved=True)

    retrieved_algorithm, commands_retrieved_algorithm = filter_invalid_copy_commands(retrieved_algorithm,
                                                                                     commands_retrieved_algorithm)

    debug = False

    if algorithm == '':
        if len(commands_retrieved_algorithm) == 1:
            if found_fillempty:
                return retrieved_algorithm
        else:
            # print('\nLog id: ', log_series.LOG_ID,
            #       '\nAlgorithm is empty and retrieved algorithm has more than one command. '
            #       'Returning retrieved algorithm:\n',
            #       retrieved_algorithm, '\n')
            return retrieved_algorithm
    else:
        algorithm, commands_algorithm, retrieved_algorithm, commands_retrieved_algorithm = process_commands(
            commands_algorithm, commands_retrieved_algorithm, regex_go, debug)

        if retrieved_algorithm == algorithm:
            return algorithm

        # Remove the go commands from the algorithm and check if like this the retrieved algorithm is equal to the algorithm
        algorithm_without_go = remove_go_commands(algorithm)
        retrieved_algorithm_without_go = remove_go_commands(retrieved_algorithm)

        if algorithm_without_go == retrieved_algorithm or algorithm_without_go == retrieved_algorithm_without_go:
            return algorithm

        has_consecutive_commands = any(commands_retrieved_algorithm[i] == commands_retrieved_algorithm[i + 1] for i in
                                       range(len(commands_retrieved_algorithm) - 1))
        has_possible_duplicate = False

        # Check the type of commands present in the retrieved algorithm
        found_go = re.search(regex_go, retrieved_algorithm)

        if not found_go and len(commands_retrieved_algorithm) == len(commands_algorithm) / 2:
            final_algorithm = ','.join(commands_algorithm)
            return final_algorithm

        sublists = group_commands(commands_retrieved_algorithm)

        final_algorithm = ''
        command_idx_to_remove = -1
        for idx, sublist in enumerate(sublists):
            # This variable represent the index of the last command in the current sublist
            last_commands_idx = len(sublist)
            command_idx_to_remove += len(sublist)
            if has_consecutive_commands:
                if has_possible_duplicate:

                    if not commands_algorithm:
                        break

                    # If there is only one command remaining in the algorithm, the current sublist is a duplicate
                    elif len(commands_algorithm) == 1:
                        command_idx_to_remove, has_possible_duplicate = remove_duplicates_from_retrieved_algorithm(
                            command_idx_to_remove, commands_retrieved_algorithm, has_possible_duplicate, idx, sublist,
                            sublists)
                        continue

                    # If the command in the sublist is different from the second command of the algorithm
                    # this means that the sublist command is a duplicate,
                    # and it needs to be removed from the commands of the retrieved algorithm
                    elif sublist[0] != commands_algorithm[last_commands_idx]:
                        command_idx_to_remove, has_possible_duplicate = remove_duplicates_from_retrieved_algorithm(
                            command_idx_to_remove, commands_retrieved_algorithm, has_possible_duplicate, idx, sublist,
                            sublists)
                        continue

                    # Particular case:
                    # The current sublist contain just a single non-go command, that is possibly a duplicate
                    # if the command in the sublist is equal to the second command of the algorithm
                    elif sublist[0] == commands_algorithm[last_commands_idx]:
                        # If there is at list another sublist after the current
                        if idx + 1 < len(sublists):
                            # If the next sublist has more than one element inside
                            if len(sublists[idx + 1]) > 1:
                                # If the command in the current sublist is equal to the second command
                                # of the next sublist (basically the non-go command)
                                if sublist[0] == sublists[idx + 1][1]:
                                    # If there are at least two more commands in the algorithm
                                    # after the current command of the sublist
                                    if len(commands_algorithm) > last_commands_idx + 1:
                                        # If the first two commands in the next sublist are equal to
                                        # the first two commands in the algorithm
                                        if (sublists[idx + 1][0], sublists[idx + 1][1]) == (
                                                commands_algorithm[last_commands_idx - 1],
                                                commands_algorithm[last_commands_idx]):
                                            # If there is not a pair of commands in the algorithm after the current pair
                                            if len(commands_algorithm) == 3:
                                                # If the non-go command in the sublist is different
                                                # from the third command in the algorithm
                                                if not (sublists[idx + 1][1]) == (
                                                        commands_algorithm[last_commands_idx + 1]):
                                                    command_idx_to_remove, has_possible_duplicate = remove_duplicates_from_retrieved_algorithm(
                                                        command_idx_to_remove, commands_retrieved_algorithm,
                                                        has_possible_duplicate, idx, sublist,
                                                        sublists)
                                                    continue
                                                else:
                                                    pass  # TODO
                                            else:
                                                # If the first two commands in the next sublist are different to
                                                # the next pair of commands in the algorithm
                                                if not (sublists[idx + 1][0], sublists[idx + 1][1]) == (
                                                        commands_algorithm[last_commands_idx + 1],
                                                        commands_algorithm[last_commands_idx + 2]):
                                                    command_idx_to_remove, has_possible_duplicate = remove_duplicates_from_retrieved_algorithm(
                                                        command_idx_to_remove, commands_retrieved_algorithm,
                                                        has_possible_duplicate, idx, sublist,
                                                        sublists)
                                                    continue

                        # Check if the sublist is the last one, if not
                        if not len(sublists) == idx + 1:
                            # Check if the last sublist command  is equal to the next sublist first command
                            # and if it is, then there can be that the next sublist correspond to a pattern,
                            # otherwise it is just a duplicate
                            if sublist[-1] == sublists[idx + 1][0]:
                                has_possible_duplicate = True
                            else:
                                has_possible_duplicate = False

                    # If there are still paint commands equal to those in the sublist,
                    # this means that there is a pattern
                    if len(commands_algorithm) > 1:
                        if sublist[0] == commands_algorithm[last_commands_idx]:
                            pass

                    if sublist[0] == commands_algorithm[last_commands_idx - 1]:
                        pass  # TODO

                else:
                    if not len(sublists) == idx + 1:
                        if sublist[-1] == sublists[idx + 1][0]:
                            has_possible_duplicate = True

            if len(commands_algorithm) == 0:
                # Merge the remaining sublists
                merged_sublists = [x for y in sublists[idx:] for x in y]
                # Check if at least one of the commands in the merged sublist is a go command
                if any(re.match(regex_go, command) for command in merged_sublists):
                    pass  # TODO
                else:
                    final_algorithm = final_algorithm[:-1]
                    return final_algorithm

            if sublist[0].startswith('paint('):

                next_sublist_command = None
                multiple_paint_pattern = None

                # If there is another sublist after the current, take the first command of the last sublist and
                # pass it to the get_paint_pattern function to understand when a pattern stops
                if idx < len(sublists) - 1:
                    if len(sublists[idx + 1]) > 1:
                        if sublists[idx + 1][1] == sublist[-1]:
                            next_sublist_command = sublists[idx + 1][0]
                else:
                    next_sublist_command = None

                # Check if the sublist is a multiple paint pattern
                if re.match(regex_pattern, sublist[0]):
                    if len(commands_algorithm) >= 4 and (commands_algorithm[1] == commands_algorithm[3]):  # there are at least 4 commands in the algorithm
                        if idx < len(sublists) - 1:  # there is another sublist after the current
                            if len(sublists[idx + 1]) > 1:  # the next sublist has more than one element inside
                                if sublists[idx + 1][1] != sublist[-1]:  # the next sublist second command is different from the last command of the current sublist
                                    if commands_algorithm[1] == sublist[0]:
                                        multiple_paint_pattern = True  # the current sublist is a multiple paint pattern
                                    else:
                                        multiple_paint_pattern = False
                                        print('Not a multiple paint pattern')
                            else:
                                if sublists[idx + 1][0] != sublist[-1]:  # the next sublist first command is different from the last command of the current sublist
                                    if commands_algorithm[1] == sublist[0]:
                                        multiple_paint_pattern = True  # the current sublist is a multiple paint pattern
                                    else:
                                        multiple_paint_pattern = False
                                        print('Not a multiple paint pattern')

                commands, command_idx = get_paint_pattern(commands_algorithm, sublist,
                                                          regex_go, regex_dot, regex_pattern,
                                                          next_sublist_command=next_sublist_command,
                                                          multiple_paint_pattern=multiple_paint_pattern)

                commands_algorithm = commands_algorithm[command_idx:]
                algorithm = ','.join(commands_algorithm)

                algorithm = remove_multiple_commas_in_algorithm(algorithm)
                commands_algorithm = get_commands_list(algorithm)

                final_algorithm += commands + ','

                final_algorithm = remove_multiple_commas_in_algorithm(final_algorithm)
            else:
                sublist_length = len(sublist)
                partial_commands_algorithm = commands_algorithm[:sublist_length]
                if sublist != partial_commands_algorithm:
                    # Missing go coordinate
                    if len(sublist) == 1:
                        element_to_add = commands_algorithm[0]
                        sublist.insert(0, element_to_add)
                    else:
                        different_index = next((index for index, (elem1, elem2) in enumerate(
                            zip(sublist, commands_algorithm)) if elem1 != elem2), None)

                        while different_index is not None:
                            equal_index = next((index for index, (elem1, elem2) in enumerate(
                                zip(sublist[different_index:], commands_algorithm[different_index:]))
                                                if elem1 == elem2), None)

                            if equal_index is None:
                                # Case 1: Missing sequence in the sublist
                                if different_index == 0:
                                    if idx < len(sublists) - 1:
                                        sublist = []  # If there is no match and there are other sublists, we assume that the current sublist can be removed
                                    else:
                                        pass
                                else:
                                    if different_index < len(sublist) - 1:  # If there are still element in the sublist
                                        partial_commands_algorithm = commands_algorithm[different_index:]
                                        partial_sublist = sublist[different_index:]

                                        boolean_subset, index = is_ordered_subset(partial_sublist,
                                                                                  partial_commands_algorithm)
                                        elements_to_add = partial_commands_algorithm[:index]

                                        # Case 1: Missing sequence in the sublist / Missing pair in the sublist
                                        if boolean_subset:
                                            for element_to_add_idx, element_to_add in enumerate(elements_to_add):
                                                sublist.insert(different_index + element_to_add_idx, element_to_add)
                                        # Case 4: There is a missing pair in commands algorithm
                                        # or it can be considered as an additional pair in the sublist
                                        else:
                                            partial_commands_algorithm = commands_algorithm[different_index - 1:]
                                            partial_sublist = sublist[different_index - 1:]

                                            temp_different_index = next((index for index, (elem1, elem2) in enumerate(
                                                zip(partial_sublist[2:], partial_commands_algorithm)) if elem1 != elem2), None)

                                            if temp_different_index is None:
                                                # Remove the additional pair in the sublist
                                                sublist.pop(different_index - 1)
                                                sublist.pop(different_index - 1)
                                            else:
                                                pass
                                    else:
                                        elements_to_add = commands_algorithm[different_index:]

                                        if not elements_to_add:
                                            pass  # TODO

                                        if len(elements_to_add) == 1 and not elements_to_add[0].startswith('go('):  # In this case there is an additional pair in the sublist
                                            break
                                        else:
                                            for element_to_add_idx, element_to_add in enumerate(elements_to_add):
                                                sublist.insert(different_index + element_to_add_idx, element_to_add)
                            else:
                                if equal_index == 1:
                                    if sublist[different_index + equal_index] == commands_algorithm[different_index + equal_index]:
                                        # Case 2: Wrong go coordinate to be changed
                                        if (len(commands_algorithm) == different_index + equal_index + 2) or (len(commands_algorithm) == 2 and idx == len(sublists) - 1) or (sublist[different_index + equal_index] != commands_algorithm[different_index + equal_index + 2]):
                                            equal_index += different_index + 1

                                            replacement_coordinate = commands_algorithm[different_index:equal_index - 1]

                                            # Remove the old coordinate in sublist and replace it with the new one
                                            sublist.pop(different_index)
                                            sublist.insert(different_index, replacement_coordinate[0])

                                        # Case 1: Missing sequence in the sublist / Missing pair in the sublist
                                        elif sublist[different_index + equal_index] == commands_algorithm[different_index + equal_index + 2]:
                                            if sublist[different_index] in commands_algorithm:
                                                equal_index += different_index
                                                index_in_commands_algorithm = commands_algorithm[different_index:].index(sublist[different_index])

                                                if (sublist[different_index + 1]) == (commands_algorithm[different_index + index_in_commands_algorithm + 1]):
                                                    equal_index += index_in_commands_algorithm
                                                    elements_to_add = commands_algorithm[different_index:equal_index - 1]

                                                    for element_to_add_idx, element_to_add in enumerate(elements_to_add):
                                                        sublist.insert(different_index + element_to_add_idx, element_to_add)
                                                else:
                                                    # Case 3: The sublist pair is wrong, substitute the sublist with the correct sequence
                                                    indices_of_occurrences = [i for i, command in
                                                                              enumerate(commands_algorithm) if
                                                                              command == sublist[1]]
                                                    last_index_of_occurrence = indices_of_occurrences[0]
                                                    for i in range(1, len(indices_of_occurrences)):
                                                        # If the difference between the current index and the previous one is greater than 1,
                                                        # then the sublist is not a pattern
                                                        if indices_of_occurrences[i] - indices_of_occurrences[
                                                            i - 1] > 2:
                                                            break
                                                        last_index_of_occurrence = indices_of_occurrences[i]

                                                    new_sublist_sequence = commands_algorithm[
                                                                           :last_index_of_occurrence + 1]

                                                    # Remove the sublist and replace it with the new sequence
                                                    sublist = new_sublist_sequence
                                            # Case 3: The sublist pair is wrong, substitute the sublist with the correct sequence
                                            else:
                                                indices_of_occurrences = [i for i, command in
                                                                          enumerate(commands_algorithm) if
                                                                          command == sublist[1]]
                                                last_index_of_occurrence = indices_of_occurrences[0]
                                                for i in range(1, len(indices_of_occurrences)):
                                                    # If the difference between the current index and the previous one is greater than 1,
                                                    # then the sublist is not a pattern
                                                    if indices_of_occurrences[i] - indices_of_occurrences[i - 1] > 2:
                                                        break
                                                    last_index_of_occurrence = indices_of_occurrences[i]

                                                new_sublist_sequence = commands_algorithm[:last_index_of_occurrence + 1]

                                                # Remove the sublist and replace it with the new sequence
                                                sublist = new_sublist_sequence
                                        else:
                                            pass  # TODO
                                    else:
                                        equal_index += different_index  # TODO
                                else:
                                    # Case 1: Missing sequence in the sublist / Missing pair in the sublist
                                    partial_commands_algorithm = commands_algorithm[different_index:]
                                    partial_sublist = sublist[different_index:]

                                    boolean_subset, index = is_ordered_subset(partial_sublist, partial_commands_algorithm)
                                    elements_to_add = partial_commands_algorithm[:index]

                                    if boolean_subset:
                                        # insert the missing sequence in the sublist
                                        for element_to_add_idx, element_to_add in enumerate(elements_to_add):
                                            sublist.insert(different_index + element_to_add_idx, element_to_add)

                                    else:
                                        # If the sublist is the last one I assume there is some error in the sublist
                                        # and I replace it with the commands algorithm
                                        if idx == len(sublists) - 1:
                                            new_sublist_sequence = commands_algorithm
                                            sublist = new_sublist_sequence
                                        else:
                                            pass  # TODO

                            sublist_length = len(sublist)
                            partial_commands_algorithm = commands_algorithm[:sublist_length]

                            different_index = next((index for index, (elem1, elem2) in enumerate(
                                zip(sublist, partial_commands_algorithm)) if elem1 != elem2), None)

                    # Remove the inserted commands from the commands_algorithm
                    commands_algorithm = commands_algorithm[sublist_length:]
                    algorithm = ','.join(commands_algorithm)
                else:
                    commands = ','.join(sublist)
                    algorithm = algorithm.replace(commands, '', 1)

                algorithm = remove_initial_comma_in_algorithm(algorithm)
                algorithm = remove_multiple_commas_in_algorithm(algorithm)

                commands_algorithm = get_commands_list(algorithm)

                if sublist:
                    final_algorithm += ','.join(sublist) + ','

        # If there are still commands available, append them
        if len(commands_algorithm) > 0:
            final_algorithm += ','.join(commands_algorithm) + ','

        final_algorithm = final_algorithm[:-1]

        return final_algorithm


def remove_duplicates_from_retrieved_algorithm(command_idx_to_remove, commands_retrieved_algorithm,
                                               has_possible_duplicate, idx, sublist, sublists):
    """
    Remove the duplicate command from the retrieved algorithm
    :param command_idx_to_remove:        the list of indices to remove from the retrieved algorithm
    :param commands_retrieved_algorithm: the list of commands of the retrieved algorithm
    :param has_possible_duplicate:       if True, the sublist has a possible duplicate
    :param idx:                          the index of the sublist
    :param sublist:                      the current sublist
    :param sublists:                     the list of sublists
    :return:                             the list of indices to remove from the retrieved algorithm
                                         and the boolean value of has_possible_duplicate
    """
    commands_retrieved_algorithm.pop(command_idx_to_remove)
    command_idx_to_remove -= 1
    # retrieved_algorithm = ','.join(commands_retrieved_algorithm)

    # Check if the sublist is the last one, if not
    if not len(sublists) == idx + 1:
        # Check if the last sublist command  is equal to the next sublist first command
        # and if it is, then there can be that the next sublist correspond to a pattern,
        # otherwise it is just a duplicate
        if sublist[-1] == sublists[idx + 1][0]:
            has_possible_duplicate = True
        else:
            has_possible_duplicate = False
    return command_idx_to_remove, has_possible_duplicate


def process_paint_differences(list1, list2, paint_list1, paint_list2, indices_list1, indices_list2):
    """
    Process the differences in the paint commands between the original and retrieved algorithms
    and return the new list of commands for the retrieved algorithm
    :param list1:         the list of commands of the original algorithm
    :param list2:         the list of commands of the retrieved algorithm
    :param paint_list1:   the list of paint commands of the original algorithm
    :param paint_list2:   the list of paint commands of the retrieved algorithm
    :param indices_list1: the list of indices corresponding to the paint commands positions in the original algorithm
    :param indices_list2: the list of indices corresponding to the paint commands positions in the retrieved algorithm
    :return:              the new list of commands for the retrieved algorithm
    """
    missing_elements_idx = []
    missing_elements = []
    extra_elements_idx = []

    # Pointers for iterating through lists 1 and 2
    i, j = 0, 0

    # Identify the missing elements in paint_list2 in relation to paint_list1
    while i < len(paint_list1) and j < len(paint_list2):
        if paint_list1[i] == paint_list2[j]:
            i += 1
            j += 1
        else:
            if not paint_list1[i] in paint_list2:
                missing_elements_idx.append(i)
                missing_elements.append(paint_list1[i])
                i += 1
            else:
                if paint_list1[i] != paint_list2[j + 1]:
                    missing_elements_idx.append(i)
                    missing_elements.append(paint_list1[i])
                    i += 1
                else:
                    j += 1

    if i < len(paint_list1):
        missing_elements_idx += list(range(len(paint_list1))[i:])
        missing_elements.extend(paint_list1[i:])

    # Pointers for iterating through lists 1 and 2
    i, j = 0, 0

    # Identify the extra elements in paint_list2 in relation to paint_list1
    while i < len(paint_list1) and j < len(paint_list2):
        if paint_list2[j] == paint_list1[i]:
            i += 1
            j += 1
        else:
            if not paint_list2[j] in paint_list1:
                extra_elements_idx.append(j)
                j += 1
            else:
                if len(paint_list2) > j + 1:
                    if paint_list2[j + 1] == paint_list1[i]:
                        extra_elements_idx.append(j)
                        j += 2
                        i += 1
                    else:
                        i += 1
                else:
                    i += 1

    if j < len(paint_list2):
        extra_elements_idx += list(range(len(paint_list2))[j:])

    new_list2 = []
    if extra_elements_idx:
        new_paint_list2 = []
        for index, element in enumerate(paint_list2):
            if index not in extra_elements_idx:
                new_paint_list2.append(element)

        # Create a mask to keep elements not in indices_to_remove
        mask = np.ones(len(paint_list2), dtype=bool)
        mask[extra_elements_idx] = False

        # Remove elements using the mask
        paint_list2 = [item for idx, item in enumerate(paint_list2) if mask[idx]]

        # Get the indices to remove from list2
        indices_to_remove = [indices_list2[idx] for idx in extra_elements_idx]

        # Step 1: Remove elements from list2 based on extra_elements_idx
        new_list2 = remove_elements_from_index(list2, indices_to_remove, new_list2)

    if not new_list2:
        new_list2 = list2[:]

    if missing_elements_idx:
        for idx, element in enumerate(missing_elements):
            paint_list2.insert(missing_elements_idx[idx], element)

        # Step 2: Add elements to list2 based on missing_elements_idx
        for idx, element in enumerate(missing_elements):
            index_to_add = indices_list1[missing_elements_idx[idx]]
            previous_index = index_to_add - 1

            while previous_index >= 0:
                previous_element = list1[previous_index]

                if previous_element in new_list2:
                    previous_element_idx = new_list2.index(previous_element)
                    new_list2.insert(previous_element_idx + 1, element)
                    break
                else:
                    previous_index = previous_index - 1

    if not np.array_equal(paint_list1, paint_list2):
        raise ValueError('Lists are not equal. \nList1: ', paint_list1, '\nList2: ', paint_list2)

    if list2 != new_list2:
        print('\nThe list of retrieved paint commands has been modified. '
              '\nold list 1\t', list1,
              '\nold list 2\t', list2,
              '\nnew list 2\t', new_list2)
    return new_list2


def process_go_differences(list2, go_list1, go_list2, indices_list2):
    """
    Process the differences in the go commands between the original and retrieved algorithms
    and return the new list of commands for the retrieved algorithm
    :param list2:         the list of commands of the retrieved algorithm
    :param go_list1:      the list of go commands of the original algorithm
    :param go_list2:      the list of go commands of the retrieved algorithm
    :param indices_list2: the list of indices corresponding to the go commands positions in the retrieved algorithm
    :return:              the new list of commands for the retrieved algorithm
    """
    extra_elements_idx = []
    elements_to_remove = []

    # Pointers for iterating through lists 1 and 2
    i, j = 0, 0

    while i < len(go_list1) and j < len(go_list2):
        if go_list1[i] == go_list2[j]:
            i += 1
            j += 1
        else:
            if not go_list2[j] in go_list1:
                extra_elements_idx.append(j)
                elements_to_remove.append(go_list2[j])
                j += 1
            else:
                i += 1

    if j < len(go_list2):
        extra_elements_idx += list(range(len(go_list2))[j:])
        elements_to_remove += go_list2[j:]

    if extra_elements_idx:
        # Get the indices to remove from list2
        # indices_to_remove = [indices_list2[idx] for idx in extra_elements_idx]

        # Remove elements from list2 based on extra_elements_idx
        new_list2 = remove_elements_from_index(list2, indices_list2, extra_elements_idx, elements_to_remove)

        if list2 != new_list2:
            print('\nThe list of retrieved go commands has been modified. '
                  '\nOriginal list: ', list2,
                  '\nNew list: ', new_list2)

        return new_list2
    else:
        return list2


def extract_command_pairs(command_list, regex_go, remove_consecutive=False):
    """
    Extracts and groups commands based on 'go' command.
    :param command_list:       the list of commands
    :param regex_go:           the regular expression for the go command
    :param remove_consecutive: if True, remove consecutive pairs
    :return:                   the list of command pairs
    """
    # Initialize the list of lists
    updated_command_list = command_list.copy()
    merged_commands = []
    i = 0
    temporary_go = None

    while i < len(command_list):
        current_command = command_list[i]
        # Check if the current command matches the "go" regex pattern
        if re.match(regex_go, current_command) and not temporary_go:
            # Merge the "go" command with the following command
            # if there is not a command after the go command just return the merged commands
            if i + 1 == len(command_list) and not temporary_go:
                return merged_commands, updated_command_list
            else:
                # If the next command is not another go command, merge the two commands
                if not re.match(regex_go, command_list[i + 1]) and not temporary_go:
                    merged_commands.append([current_command, command_list[i + 1]])
                    i += 2  # Skip the next command since it has been merged
                # If the next command is another go command, skip it for the moment and save it in a temporary variable,
                # check the element after that, and if it is not a go command, merge the two commands together
                else:
                    if not re.match(regex_go, command_list[i + 2]) and not temporary_go:
                        temporary_go = command_list[i + 1]
                        merged_commands.append([current_command, command_list[i + 2]])
                        # Swap the elements at the given index and the next index
                        updated_command_list[i + 1], updated_command_list[i + 2] = updated_command_list[i + 2], \
                            updated_command_list[i + 1]
                        i += 3
        elif not re.match(regex_go, current_command) and temporary_go:
            merged_commands.append([temporary_go, current_command])
            temporary_go = None
            i += 1
        elif re.match(regex_go, current_command) and temporary_go:
            raise ValueError('The temporary go command is not None.')
        else:
            # If it's not a "go" command, keep it as is
            merged_commands.append([current_command])
            i += 1

    if remove_consecutive:
        new_merged_commands = remove_consecutive_pairs(merged_commands)
        updated_command_list = [item for sublist in new_merged_commands for item in sublist]
    else:
        new_merged_commands = merged_commands

    return new_merged_commands, updated_command_list


def filter_pair_list(pairs_list):
    """
    Filter the consecutive paint commands and track the indices of the paint commands
    :param pairs_list: the list of pair commands
    :return:           the filtered list of pair commands and the list of indices
    """
    if len(pairs_list) == 1:
        return pairs_list, [0], [0]
    elif len(pairs_list) == 0:
        raise ValueError('The list of pairs is empty.')
    else:
        filtered_list = []
        filtered_indexes = []
        index = 0
        for i, pair_command in enumerate(pairs_list):
            command = pair_command[-1]
            # Add the first command to the list by default
            if i == 0:
                filtered_list.append(pair_command)
                filtered_indexes.append(i)
                if len(pair_command) > 1:
                    filtered_indexes.append(i + 1)
                index += len(pair_command)
            else:
                previous_command = pairs_list[i - 1][-1]
                if command != previous_command:
                    filtered_list.append(pair_command)
                    filtered_indexes.append(index)
                    if len(pair_command) > 1:
                        filtered_indexes.append(index + 1)
                index += len(pair_command)

        indexes = np.arange(len(pairs_list)).tolist()

        return filtered_list, indexes, filtered_indexes


def verify_equal_pairs(pairs_list1, pairs_list2):
    """
    Verify if the two lists of pairs are equal
    :param pairs_list1:         the list of pairs of the original algorithm
    :param pairs_list2:         the list of pairs of the retrieved algorithm
    :return:                    True if the two lists of pairs are equal, False otherwise
    """
    equal_pairs = False
    # Check if the two lists have the same length
    if len(pairs_list1) == len(pairs_list2):
        # Take just the last element of each sublist in each list and verify if they are equal
        if [sublist[-1] for sublist in pairs_list1] == [sublist[-1] for sublist in pairs_list2]:
            equal_pairs = True
    return equal_pairs


def check_and_return_result(original_commands_list2, commands_list1, new_commands_list2, regex_go):
    """
    Check if the new list of retrieved commands is equal to the original one
    and return the new retrieved algorithm and the new list of retrieved commands
    if they are equal, None otherwise
    :param original_commands_list2: the original list of retrieved commands
    :param commands_list1:          the list of commands of the original algorithm
    :param new_commands_list2:      the new list of retrieved commands
    :param regex_go:                the regular expression for the go command
    :return:                        the boolean result that is False if the two lists are not equal, True otherwise
                                    and all the other parameters needed
    """
    result = False

    # if original_commands_list2 != new_commands_list2:
    #     print('\nThe list of retrieved commands has been modified. '
    #           '\nOriginal list:\t', original_commands_list2,
    #           '\nNew list:\t', new_commands_list2)

    pairs_list1, new_commands_list1 = extract_command_pairs(commands_list1, regex_go, remove_consecutive=True)
    pairs_list2, new_commands_list2 = extract_command_pairs(new_commands_list2, regex_go, remove_consecutive=True)
    equal_pairs = verify_equal_pairs(pairs_list1, pairs_list2)

    if equal_pairs:
        result = True
        retrieved_algorithm = ','.join(new_commands_list2)
        return result, retrieved_algorithm, new_commands_list1, new_commands_list2, None, None, None, None, None, None

    filtered_pairs_list1, commands_list1_indexes, pairs_list1_indexes = filter_pair_list(pairs_list1)
    filtered_pairs_list2, commands_list2_indexes, pairs_list2_indexes = filter_pair_list(pairs_list2)

    equal_pairs = verify_equal_pairs(filtered_pairs_list1, filtered_pairs_list2)

    if equal_pairs:
        result = True
        retrieved_algorithm = ','.join(new_commands_list2)
        return result, retrieved_algorithm, new_commands_list1, new_commands_list2, None, None, None, None, None, None

    retrieved_algorithm = ','.join(new_commands_list2)

    return (result,
            retrieved_algorithm,
            new_commands_list1, new_commands_list2,
            filtered_pairs_list1, filtered_pairs_list2,
            commands_list1_indexes, commands_list2_indexes,
            pairs_list1_indexes, pairs_list2_indexes)


def process_commands(commands_list1, commands_list2, regex_go, debug=False):
    """
    Process the commands of the two lists and return the new list of retrieved commands
    :param commands_list1: the list of commands of the original algorithm
    :param commands_list2: the list of commands of the retrieved algorithm
    :param regex_go:       the regular expression for the go command
    :param debug:          if True, the breakpoint is activated
    :return:               the new retrieved algorithm and the new list of retrieved commands
    """
    # Original command list 2
    original_commands_list2 = commands_list2.copy()

    # Extract the list of go commands from the commands lists
    go_commands_list1 = [element for element in commands_list1 if re.match(regex_go, element)]
    go_commands_list2 = [element for element in commands_list2 if re.match(regex_go, element)]

    # Find indices of go commands in the original lists
    # go_indexes2 = np.where(np.isin(commands_list2, go_commands_list2))[0].tolist()

    if go_commands_list1 == go_commands_list2:
        algorithm = ','.join(commands_list1)
        # We assume that the retrieved algorithm is equal to the original one so we return the original algorithm twice
        return algorithm, commands_list1, algorithm, commands_list1

    # if not go_commands_list1 == go_commands_list2 and len(go_commands_list2) > 0:
    #     # Process the differences between the two lists of go commands and return the new list of retrieved commands
    #     # Extra go commands in list 2 are removed
    #     new_commands_list2 = process_go_differences(commands_list2, go_commands_list1, go_commands_list2, go_indexes2)
    # else:
    new_commands_list2 = commands_list2

    # Remove the go commands from the algorithm and check if like this the retrieved algorithm is equal to the algorithm
    algorithm = ','.join(commands_list1)
    retrieved_algorithm = ','.join(new_commands_list2)

    algorithm_without_go = remove_go_commands(algorithm)
    retrieved_algorithm_without_go = remove_go_commands(retrieved_algorithm)

    if algorithm_without_go == retrieved_algorithm or algorithm_without_go == retrieved_algorithm_without_go:
        # We assume that the retrieved algorithm is equal to the original one so we return the original algorithm twice
        return algorithm, commands_list1, algorithm, commands_list1

    (result,
     retrieved_algorithm,
     commands_list1, new_commands_list2,
     filtered_pairs_list1, filtered_pairs_list2,
     _, _,
     pairs_list1_indexes, pairs_list2_indexes) = check_and_return_result(original_commands_list2,
                                                                         commands_list1,
                                                                         new_commands_list2,
                                                                         regex_go)

    if result:
        algorithm = ','.join(commands_list1)
        return algorithm, commands_list1, retrieved_algorithm, new_commands_list2

    if len(filtered_pairs_list2) == 1 and len(filtered_pairs_list1) > 2:
        algorithm = ','.join(commands_list1)
        # We assume that the retrieved algorithm is equal to the original one so we return the original algorithm twice
        return algorithm, commands_list1, algorithm, commands_list1

    # Add the missing commands to the retrieved algorithm Step (A)
    retrieved_algorithm, new_commands_list2 = add_missing_commands_step_a(filtered_pairs_list1, filtered_pairs_list2,
                                                                          pairs_list2_indexes, new_commands_list2,
                                                                          debug)
    (result,
     retrieved_algorithm,
     commands_list1, new_commands_list2,
     filtered_pairs_list1, filtered_pairs_list2,
     _, commands_list2_indexes,
     pairs_list1_indexes, pairs_list2_indexes) = check_and_return_result(original_commands_list2,
                                                                         commands_list1,
                                                                         new_commands_list2,
                                                                         regex_go)

    if result:
        algorithm = ','.join(commands_list1)
        return algorithm, commands_list1, retrieved_algorithm, new_commands_list2

    # Remove the extra commands from the retrieved algorithm Step (A)
    retrieved_algorithm, new_commands_list2 = remove_extra_commands_step_a(filtered_pairs_list1, filtered_pairs_list2,
                                                                           pairs_list2_indexes, new_commands_list2,
                                                                           debug)
    (result,
     retrieved_algorithm,
     commands_list1, new_commands_list2,
     filtered_pairs_list1, filtered_pairs_list2,
     _, _,
     pairs_list1_indexes, pairs_list2_indexes) = check_and_return_result(original_commands_list2,
                                                                         commands_list1,
                                                                         new_commands_list2,
                                                                         regex_go)

    if result:
        algorithm = ','.join(commands_list1)
        return algorithm, commands_list1, retrieved_algorithm, new_commands_list2

    # Add the missing commands to the retrieved algorithm Step (B)
    retrieved_algorithm, new_commands_list2 = add_missing_commands_step_b(filtered_pairs_list1, filtered_pairs_list2,
                                                                          new_commands_list2,
                                                                          pairs_list2_indexes,
                                                                          debug)
    (result,
     retrieved_algorithm,
     commands_list1, new_commands_list2,
     filtered_pairs_list1, filtered_pairs_list2,
     _, commands_list2_indexes,
     pairs_list1_indexes, pairs_list2_indexes) = check_and_return_result(original_commands_list2,
                                                                         commands_list1,
                                                                         new_commands_list2,
                                                                         regex_go)

    if result:
        algorithm = ','.join(commands_list1)
        return algorithm, commands_list1, retrieved_algorithm, new_commands_list2
    else:
        algorithm = ','.join(commands_list1)
        # We assume that the retrieved algorithm is wrong and make it equal to the original one
        return algorithm, commands_list1, algorithm, commands_list1


def add_missing_commands_step_a(f_pairs_list1, f_pairs_list2, pairs_list2_indexes, commands_list2, debug=False):
    """
    Add the missing commands to the retrieved algorithm.
    :param f_pairs_list1:          the filtered list of pairs of commands of the algorithm
    :param f_pairs_list2:          the filtered list of pairs of commands of the retrieved algorithm
    :param pairs_list2_indexes:    the list of pairs of commands of the retrieved algorithm without consecutive equal commands
    :param commands_list2:         the list of commands of the retrieved algorithm
    :param debug:                  if True, the breakpoint is triggered
    :return:                       the retrieved algorithm and the list of commands of the retrieved algorithm
    """
    # Step (A): For each command in the list 1, check if there is a correspondence in the list 2,
    # if there is not, then the command is for sure missing
    missing_elements_idx = []
    missing_elements = []
    l1 = 0
    l2 = 0
    last_l2 = 0

    while l1 < len(f_pairs_list1) and l2 < len(f_pairs_list2):
        sublist1 = f_pairs_list1[l1]
        element_sublist1 = sublist1[-1]
        found_missing = False

        while l2 < len(f_pairs_list2):
            sublist2 = f_pairs_list2[l2]
            element_sublist2 = sublist2[-1]

            if element_sublist1 == element_sublist2 or element_sublist1 in missing_elements:
                found_missing = False
                break
            else:
                found_missing = True
                l2 += 1

        if found_missing is True:
            # To understand the correct index of the missing element take all sublists till the current one
            # Than merge the lists and count the elements, the result is the index of the missing element
            index_of_last_non_missing_element = len([element for sublist in f_pairs_list2[:last_l2] for element in sublist]) - 1

            # if index_of_last_non_missing_element == - 1 this means that the missing element is the first one

            missing_elements.append(element_sublist1)

            # If the index of the last non-missing element is the last in the list
            if index_of_last_non_missing_element == len(pairs_list2_indexes) - 1:
                missing_elements_idx.append(len(commands_list2) + len(missing_elements_idx))

            #  If the missing element is not the last in the list, check if the next element is consecutive
            elif index_of_last_non_missing_element < len(pairs_list2_indexes) - 1:
                # If the current sublist1 has two elements,
                if len(sublist1) == 2:
                    # and the next sublist2 has two elements, one coordinate and one command,
                    if len(sublist2) == 2:
                        # insert the command pattern after the coordinate
                        index_of_last_non_missing_element += 1
                    else:
                        # Otherwise, do nothing
                        pass

                if index_of_last_non_missing_element == -1 and len(missing_elements_idx) > 0:
                    add_missing_element_at_index = missing_elements_idx[-1] + 1
                    indices = list(range(add_missing_element_at_index, add_missing_element_at_index + 1))
                else:
                    if index_of_last_non_missing_element == -1:
                        add_missing_element_at_index = 0
                    else:
                        add_missing_element_at_index = pairs_list2_indexes[index_of_last_non_missing_element] + 1

                    next_index = pairs_list2_indexes[index_of_last_non_missing_element + 1]

                    if next_index != add_missing_element_at_index:
                        print('Check')
                        if 'copy' in element_sublist1:
                            indices = list(range(next_index,
                                                 next_index + len(missing_elements_idx) + 1))
                        else:
                            indices = list(range(add_missing_element_at_index,
                                                 add_missing_element_at_index + len(missing_elements_idx) + 1))  # +1 is the length of the added command
                    else:
                        indices = list(range(add_missing_element_at_index,
                                             add_missing_element_at_index + len(missing_elements_idx) + 1))  # +1 is the length of the added command

                missing_elements_idx += indices

            else:
                pass  # TODO

        l1 += 1
        if found_missing is False:
            last_l2 += 1
        else:
            last_l2 += 0
        l2 = 0

    missing_elements, missing_elements_idx = append_remaining_elements_to_missing(commands_list2, f_pairs_list1, l1,
                                                                                  missing_elements, missing_elements_idx)

    # Remove elements from list2 based on extra_elements_idx
    if len(missing_elements_idx) > 0:
        new_commands_list2 = add_elements_from_index(commands_list2, missing_elements_idx, missing_elements)
    else:
        new_commands_list2 = commands_list2

    retrieved_algorithm = ','.join(new_commands_list2)

    return retrieved_algorithm, new_commands_list2


def append_remaining_elements_to_missing(commands_list2, f_pairs_list1, l1, missing_elements, missing_elements_idx):
    """
    Append the remaining elements to the missing elements list and compute the indices
    :param commands_list2:       the list of commands of the retrieved algorithm
    :param f_pairs_list1:        the filtered list of pairs of commands of the algorithm
    :param l1:                   the index of the last element in the list 1
    :param missing_elements:     the list of missing elements
    :param missing_elements_idx: the list of indices corresponding to the missing elements
    :return:                     the list of missing elements and the list of indices corresponding to the missing elements
    """
    # If there are still elements in the first list, add them to the missing elements and compute indices
    if l1 <= len(f_pairs_list1) - 1:
        # Take all sublists from the current one and merge them, obtaining the list of remaining elements
        remaining_elements = [element for sublist in f_pairs_list1[l1:] for element in sublist]
        # Add the remaining elements to the missing elements
        missing_elements += remaining_elements

        # Compute the missing indices
        missing_indices = list(range(len(commands_list2) + len(missing_elements_idx),
                                     len(commands_list2) + len(missing_elements_idx) + len(remaining_elements)))
        missing_elements_idx += missing_indices

    return missing_elements, missing_elements_idx


def add_missing_commands_step_b(f_pairs_list1, f_pairs_list2, commands_list2, pairs_list2_indexes, debug=False):
    """
    Add the missing commands to the retrieved algorithm.
    :param f_pairs_list1:          the filtered list of pairs of commands of the algorithm
    :param f_pairs_list2:          the filtered list of pairs of commands of the retrieved algorithm
    :param commands_list2:         the list of commands of the retrieved algorithm
    :param pairs_list2_indexes:    the list of pairs of commands of the retrieved algorithm without consecutive equal commands
    :param debug:                  if True, the breakpoint is triggered
    :return:                       the retrieved algorithm and the list of commands of the retrieved algorithm
    """
    # Step (B): Identify the other missing commands in f_pairs_list2 in relation to f_pairs_list1
    missing_elements_idx = []
    missing_elements = []
    i, j = 0, 0

    temp_missing_elements_idx = []
    temp_missing_elements = []

    # Identify the missing elements in paint_list2 in relation to paint_list1
    while i < len(f_pairs_list1) and j < len(f_pairs_list2):
        # If the pair is the same, move to the next pair
        if f_pairs_list1[i] == f_pairs_list2[j]:  # match = True
            i += 1
            j += 1
        elif len(f_pairs_list2[j]) == 1 and f_pairs_list1[i][-1] == f_pairs_list2[j][-1]:
            i += 1
            j += 1
        else:
            element_l2 = f_pairs_list2[j][-1]
            for pair_idx, pair in enumerate(f_pairs_list1[i:]):
                element_l1 = pair[-1]

                boolean_same_pair = f_pairs_list2[j] == pair and len(f_pairs_list2[j]) == 2
                boolean_same_pair_element = element_l2 == element_l1 and len(f_pairs_list2[j]) == 1
                if boolean_same_pair or boolean_same_pair_element:  # match = True
                    index_of_last_non_missing_element = len([element for sublist in f_pairs_list2[:j] for element in sublist]) - 1

                    if index_of_last_non_missing_element == -1:
                        add_missing_element_at_index = 0
                    else:
                        add_missing_element_at_index = pairs_list2_indexes[index_of_last_non_missing_element] + 1

                    pairs_list1_till_previous = f_pairs_list1[i:pair_idx + i]

                    # List with the missing elements
                    merged_list = [element for sublist in pairs_list1_till_previous for element in sublist]
                    num_missing_elements = len(merged_list)

                    # If the number of missing elements is higher than the difference of elements between the two pair lists
                    # ignore the missing elements
                    len_f_pairs_list1 = len([element for sublist in f_pairs_list1 for element in sublist])
                    len_f_pairs_list2 = len([element for sublist in f_pairs_list2 for element in sublist])
                    difference = abs(len_f_pairs_list1 - len_f_pairs_list2)
                    if num_missing_elements > difference:
                        i += 1
                        j += 1
                        break
                    else:
                        indices = list(range(add_missing_element_at_index + len(temp_missing_elements_idx),
                                             add_missing_element_at_index + len(temp_missing_elements_idx) + num_missing_elements))

                        if indices == []:
                            indices = list(range(i + 1, num_missing_elements + i + 1))

                        temp_missing_elements_idx += indices
                        temp_missing_elements += merged_list

                        i += len(pairs_list1_till_previous) + 1
                        j += 1
                        break

            # This is a very special and uncommon case,
            if boolean_same_pair is False and boolean_same_pair_element is False:
                # I assume that there are just problems with the coordinates
                if j == len(f_pairs_list2) - 1 and i < len(f_pairs_list1) - 1:
                    i += 1
                    j += 1
                # I assume that if this happens maybe the algorithm is wrong
                # Just ignore it for the moment and let's see what happen later
                else:
                    temp_missing_elements_idx = missing_elements_idx.copy()
                    temp_missing_elements = missing_elements.copy()
                    j += 1
                    i = j
            else:
                missing_elements_idx = temp_missing_elements_idx.copy()
                missing_elements = temp_missing_elements.copy()

    missing_elements, missing_elements_idx = append_remaining_elements_to_missing(commands_list2, f_pairs_list1, i,
                                                                                  missing_elements, missing_elements_idx)

    # Remove elements from list2 based on extra_elements_idx
    if len(missing_elements_idx) > 0:
        new_commands_list2 = add_elements_from_index(commands_list2, missing_elements_idx, missing_elements)
    else:
        new_commands_list2 = commands_list2

    retrieved_algorithm = ','.join(new_commands_list2)

    return retrieved_algorithm, new_commands_list2


def remove_extra_commands_step_a(f_pairs_list1, f_pairs_list2, pairs_list2_indexes, commands_list2, debug=False):
    """
    Remove the extra commands from the retrieved algorithm.
    :param f_pairs_list1:       the list of pairs of commands of the algorithm
    :param f_pairs_list2:       the list of pairs of commands of the retrieved algorithm
    :param pairs_list2_indexes: the list of pairs of commands of the retrieved algorithm without consecutive equal commands
    :param commands_list2:      the list of commands of the retrieved algorithm
    :param debug:               if True, the breakpoint is triggered
    :return:                    the retrieved algorithm and the list of commands of the retrieved algorithm
    """
    # Step (A): For each command in the list 2, check if there is a correspondence in the list 1,
    # if there is not, then the command is for sure extra
    extra_elements_idx = []
    extra_elements = []
    k = 0
    while k < len(f_pairs_list2):
        element_sublist2 = f_pairs_list2[k][-1]
        found_extra = False
        for sublist1 in f_pairs_list1:
            element_sublist1 = sublist1[-1]
            if element_sublist2 == element_sublist1:
                found_extra = False
                break
            else:
                found_extra = True
        if found_extra is True:
            # To understand the correct index of the extra element take all sublists till the current one
            # Than merge the lists and count the elements, the result is the index of the missing element
            supplement_idx = 0

            if len(f_pairs_list2[k]) == 2:
                if f_pairs_list2[k][0] == f_pairs_list1[k][1]:
                    supplement_idx += 1
                else:
                    supplement_idx += 2

            merged_list = [element for sublist in f_pairs_list2[:k] for element in sublist]

            if supplement_idx == 1:
                merged_list.append(f_pairs_list2[k][0])

            if supplement_idx == 2:
                if len(merged_list) == len(commands_list2):
                    pass  # TODO
                else:
                    extra_elements_idx += [len(merged_list), len(merged_list)+1]
                    extra_elements += f_pairs_list2[k]
            else:
                if len(merged_list) == len(commands_list2):
                    extra_element_idx = -1
                else:
                    extra_element_idx = len(merged_list)

                extra_elements_idx.append(extra_element_idx)
                extra_elements.append(f_pairs_list2[k][-1])

        k += 1

    # Remove elements from list2 based on extra_elements_idx
    if len(extra_elements_idx) > 0:
        new_commands_list2 = remove_elements_from_index(commands_list2, pairs_list2_indexes,
                                                        extra_elements_idx, extra_elements)
    else:
        new_commands_list2 = commands_list2

    retrieved_algorithm = ','.join(new_commands_list2)

    return retrieved_algorithm, new_commands_list2


def remove_extra_commands_step_b(f_pairs_list1, f_pairs_list2, pairs_list2_indexes, commands_list2, debug=False):
    """
    Remove the extra commands from the retrieved algorithm.
    :param f_pairs_list1:         the list of pairs of commands of the algorithm
    :param f_pairs_list2:         the list of pairs of commands of the retrieved algorithm
    :param pairs_list2_indexes:   the list of pairs of commands of the retrieved algorithm without consecutive equal commands
    :param commands_list2:        the list of commands of the retrieved algorithm
    :param debug:                 if True, the breakpoint is triggered
    :return:                      the retrieved algorithm and the list of commands of the retrieved algorithm
    """
    extra_elements_idx = []
    i, j = 0, 0
    while i < len(f_pairs_list1) and j < len(f_pairs_list2):
        sublist1 = f_pairs_list1[i]
        sublist2 = f_pairs_list2[j]
        # If both the go and non go commands correspond we are sure that the command pair in list 2 is not extra,
        # so we can move to the next pair
        if sublist1 == sublist2:
            i += 1
            j += 1
        else:
            # Compare the non go command with the non go command in list 1,
            # if they are equal, the  move to the next pair in list 1
            if sublist1[-1] == sublist2[-1]:
                i += 1
                j += 1
            else:
                # There are consecutive equal commands in list 1
                if sublist1[-1] == f_pairs_list1[i + 1][-1]:
                    i += 1
                else:
                    i += 1

    if j < len(f_pairs_list2):
        extra_elements_idx += list(range(len(f_pairs_list2))[j:])

    # Remove elements from list2 based on extra_elements_idx
    if len(extra_elements_idx) > 0:
        new_commands_list2 = remove_elements_from_index(commands_list2, pairs_list2_indexes, extra_elements_idx)
    else:
        new_commands_list2 = commands_list2

    retrieved_algorithm = ','.join(new_commands_list2)
    return retrieved_algorithm, new_commands_list2


def process_missing_commands(commands_list1, commands_list2, regex_go, regex_wrong_paint, debug):
    """
    Process the missing commands in the retrieved algorithm:
    the extra go commands are removed from the retrieved algorithm,
    the missing paint commands are added to the retrieved algorithm and
    the extra paint commands are removed from the retrieved algorithm.

    :param commands_list1:   the list of commands of the algorithm
    :param commands_list2:   the list of commands of the retrieved algorithm
    :param regex_go:          the regular expression for the go command
    :param regex_wrong_paint: the regular expression for the empty paint command
    :return:                  the retrieved algorithm and the list of commands of the retrieved algorithm
    """
    # Extract the list of go commands from the commands lists
    go_commands_list1 = [element for element in commands_list1 if re.match(regex_go, element)]
    go_commands_list2 = [element for element in commands_list2 if re.match(regex_go, element)]

    # Find indices of go commands in the original lists
    go_indexes2 = np.where(np.isin(commands_list2, go_commands_list2))[0].tolist()

    if go_commands_list1 == go_commands_list2:
        retrieved_algorithm = ','.join(commands_list1)
        return retrieved_algorithm, commands_list1

    if not go_commands_list1 == go_commands_list2 and len(go_commands_list2) > 0:
        # Process the differences between the two lists of go commands and return the new list of retrieved commands
        # Extra go commands in list 2 are removed, missing go commands in list 2 are not added
        commands_list2 = process_go_differences(commands_list2, go_commands_list1, go_commands_list2, go_indexes2)

    # Extract the list of paint commands from the commands lists
    paint_commands_list1 = [element for element in commands_list1 if not re.match(regex_go, element)]
    paint_commands_list2 = [element for element in commands_list2 if not re.match(regex_go, element)]

    # Find indices of paint commands in the original lists
    paint_indexes1 = np.where(np.isin(commands_list1, paint_commands_list1))[0].tolist()
    paint_indexes2 = np.where(np.isin(commands_list2, paint_commands_list2))[0].tolist()

    # Remove equal consecutive elements from the lists
    filtered_paint_commands_list1, indices1 = filter_and_track_indexes(paint_commands_list1)
    filtered_paint_commands_list2, indices2 = filter_and_track_indexes(paint_commands_list2)

    filtered_paint_indexes1 = keep_elements_from_index(paint_indexes1, indices1)
    filtered_paint_indexes2 = keep_elements_from_index(paint_indexes2, indices2)

    if not filtered_paint_commands_list1 == filtered_paint_commands_list2:
        if len(commands_list1) == len(commands_list2) and \
                re.match(regex_wrong_paint, ','.join(paint_commands_list2)):
            return ','.join(commands_list2), commands_list1
        # Process the differences between the two lists of paint commands and return the new list of retrieved commands
        commands_list2 = process_paint_differences(commands_list1, commands_list2,
                                                   filtered_paint_commands_list1, filtered_paint_commands_list2,
                                                   filtered_paint_indexes1, filtered_paint_indexes2)

    retrieved_algorithm = ','.join(commands_list2)

    return retrieved_algorithm, commands_list2


def filter_invalid_go_commands(algorithm, commands, log_id):
    """
    Filter the invalid go commands from the algorithm and commands list
    :param algorithm: the algorithm
    :param commands:  the list of commands
    :param log_id:    the log id
    :return:          the filtered algorithm and commands list
    """
    coordinates_to_check = ['a1', 'a2', 'a5', 'a6',
                            'b1', 'b2', 'b5', 'b6',
                            'e1', 'e2', 'e5', 'e6',
                            'f1', 'f2', 'f5', 'f6']

    # Check if the algorithm contains the go commands with invalid coordinates
    invalid_go_algorithm = any(param in c for param in coordinates_to_check for c in commands if c.startswith('go('))

    # Remove the go commands with invalid coordinates
    if invalid_go_algorithm:
        # Print the invalid commands
        # print('\nInvalid GO() command for log id: ', log_id, '\n\talgorithm: ', algorithm,
        #       '\n\t->', [c for c in commands if any(param in c for param in coordinates_to_check)])
        filtered_commands = [c if not any(param in c for param in coordinates_to_check) else 'go(c1)' for c in commands]

        algorithm = ','.join(filtered_commands)

        return algorithm, filtered_commands
    else:
        return algorithm, commands


def remove_consecutive_pairs(pairs):
    """
    Remove the consecutive equal pairs from the pairs list
    :param pairs: the list of pairs
    :return:      the filtered pairs list
    """
    filtered_pairs = []
    prev_pair = None

    for pair in pairs:
        # If the pair is different from the previous one, add it to the filtered pairs list
        if pair != prev_pair:
            if prev_pair and 'fill_empty' in pair[-1] and 'fill_empty' in prev_pair[-1] and pair[-1] == prev_pair[-1]:
                pass
            else:
                filtered_pairs.append(pair)

        # Update the previous pair
        prev_pair = pair

    return filtered_pairs


def remove_consecutive_go(commands, retrieved=False):
    """
    Remove the consecutive go commands from the commands list keeping only the last one
    :param commands: the list of commands
    :param retrieved: if True, print the commands that have been removed
    :return:         the filtered algorithm and commands list
    """
    filtered_commands = []
    saved_go = None

    for command in commands:
        if command.startswith('go('):
            saved_go = command
        else:
            if saved_go:
                filtered_commands.append(saved_go)
                filtered_commands.append(command)
                saved_go = None
            else:
                filtered_commands.append(command)

    #  If the commands list ends with a go command, append it to the filtered_commands list
    if saved_go:
        filtered_commands.append(saved_go)

    algorithm = ','.join(filtered_commands)

    # if retrieved:
    #     if commands != filtered_commands:
    #         print('\nRemoved consecutive go commands from the retrieved algorithm:\n',
    #               commands, '\n->\n', filtered_commands)

    return algorithm, filtered_commands


def remove_go_commands(algorithm):
    """
    Remove the go commands from the algorithm
    :param algorithm: the algorithm
    :return:          the filtered algorithm
    """
    filtered_algorithm = re.sub(r'go\(.*?\),', '', algorithm)
    filtered_algorithm = remove_initial_comma_in_algorithm(filtered_algorithm)
    filtered_algorithm = remove_multiple_commas_in_algorithm(filtered_algorithm)
    filtered_algorithm = remove_final_comma_in_algorithm(filtered_algorithm)

    return filtered_algorithm


def group_commands(commands_retrieved_algorithm):
    """
    Group the commands in the commands_retrieved_algorithm
    :param commands_retrieved_algorithm: the commands of the retrieved algorithm
    :return:                             the sublists of commands
    """
    sublists = []
    current_sublist = []

    for command in commands_retrieved_algorithm:
        if not current_sublist:
            if command.startswith('paint(') or command.startswith('fill_empty('):
                sublists.append([command])
            elif command.startswith('go('):
                current_sublist.append(command)
            elif command.startswith('copy('):
                current_sublist.append(command)
            elif command.startswith('mirror('):
                current_sublist.append(command)
            else:
                raise ValueError('Invalid command: ', command)
        else:
            if command.startswith('paint(') or command.startswith('fill_empty('):
                if not current_sublist[-1].startswith('go('):
                    sublists.append(current_sublist)
                    current_sublist = []
                    sublists.append([command])
                else:
                    current_sublist.append(command)
            elif command.startswith('go('):
                current_sublist.append(command)
            elif command.startswith('copy('):
                current_sublist.append(command)
            elif command.startswith('mirror('):
                current_sublist.append(command)
            else:
                raise ValueError('Invalid command: ', command)

    if current_sublist:
        sublists.append(current_sublist)

    return sublists


def get_paint_pattern(commands_algorithm, commands_retrieved_algorithm, regex_go, regex_dot, regex_pattern,
                      next_sublist_command=None, multiple_paint_pattern=None):
    """
    Get the paint pattern from the commands_algorithm and commands_retrieved_algorithm
    :param commands_algorithm:           the list of commands of the algorithm
    :param commands_retrieved_algorithm: the list of commands of the retrieved algorithm
    :param regex_go:                     the regular expression for the go command
    :param regex_dot:                    the regular expression for the dot command
    :param regex_pattern:                the regular expression for the pattern command
    :param next_sublist_command:         the next sublist command
    :param multiple_paint_pattern:       if True, the paint pattern is multiple
    :return:                             the paint pattern and the index of the next command
    """
    go_coordinates_list = []
    paint_colours_list = []
    c_idx = 0

    missing_algorithm = ''

    for paint_command in commands_retrieved_algorithm:
        paint_colours_list.append(paint_command.split('(')[1].split(')')[0])
        paint_command_dot_match = re.match(regex_dot, paint_command)
        paint_command_pattern_match = re.match(regex_pattern, paint_command)
        if multiple_paint_pattern is not None:
            if multiple_paint_pattern:
                coordinates = ''
                while c_idx < len(commands_algorithm):
                    command = commands_algorithm[c_idx]
                    command_pattern_match = re.match(regex_pattern, command)
                    if command.startswith('go('):
                        if next_sublist_command is not None and command == next_sublist_command:
                            break
                        no_next_command = len(commands_algorithm) == c_idx + 1
                        if no_next_command:
                            break
                        next_command = commands_algorithm[c_idx + 1]
                        if paint_command in next_command:
                            go_parameter = command.split('(')[1].split(')')[0]

                            coordinates += go_parameter + ','
                            c_idx += 1
                        else:
                            # c_idx don't need to be updated
                            break
                    elif command_pattern_match:
                        c_idx += 1
                    elif command.startswith('paint('):
                        break
                    elif command.startswith('fill_empty('):
                        break
                    elif command.startswith('copy('):
                        break
                    elif command.startswith('mirror('):
                        break
                    else:
                        raise ValueError('Invalid command: ', command)

                # Remove last comma from coordinates
                if next_sublist_command is not None and not coordinates:
                    return "", c_idx
                elif not coordinates:
                    raise ValueError('Invalid coordinates: ', coordinates)
                else:
                    if coordinates[-1] == ',':
                        coordinates = coordinates[:-1]
                    else:
                        raise ValueError('Invalid end for coordinates: ', coordinates)

                # Append coordinates to the list
                go_coordinates_list.append(coordinates)

                # Create the final algorithm from the coordinates and colours
                if len(paint_colours_list) != 1:
                    raise ValueError('Invalid paint colours list: ', paint_colours_list)

                for go_coordinates, paint_colours in zip(go_coordinates_list, paint_colours_list):
                    # final_algorithm = ','.join([f'copy({{{paint_colours}}}{{{go_coordinates}}})'])
                    color = re.search(r'{(.*?)}', paint_colours).group(1)
                    repetitions = int(re.search(r'{.*},(\d+)', paint_colours).group(1))
                    direction = re.search(r'{.*},\d+,(.*)', paint_colours).group(1)

                    # Update the coordinates based on the direction and repetitions
                    old_coordinates = go_coordinates.split(',')
                    new_coordinates = []
                    for i, coordinate in enumerate(old_coordinates):
                        new_coordinates.append(coordinate)

                        coordinate_letter = coordinate[0]
                        coordinate_number = int(coordinate[1:])
                        for r in range(1, repetitions):
                            if direction == 'right':
                                new_coordinates.append(coordinate_letter + str(coordinate_number + r))
                            elif direction == 'left':
                                new_coordinates.append(coordinate_letter + str(coordinate_number - r))
                            elif direction == 'up':
                                new_coordinates.append(chr(ord(coordinate_letter) + r) + str(coordinate_number))
                            elif direction == 'down':
                                new_coordinates.append(chr(ord(coordinate_letter) - r) + str(coordinate_number))
                            elif direction == 'diagonal up right':
                                new_coordinates.append(chr(ord(coordinate_letter) + r) + str(coordinate_number + r))
                            elif direction == 'diagonal up left':
                                new_coordinates.append(chr(ord(coordinate_letter) + r) + str(coordinate_number - r))
                            elif direction == 'diagonal down right':
                                new_coordinates.append(chr(ord(coordinate_letter) - r) + str(coordinate_number + r))
                            elif direction == 'diagonal down left':
                                new_coordinates.append(chr(ord(coordinate_letter) - r) + str(coordinate_number - r))
                            elif 'square' in direction:
                                new_commands_algorithm = commands_algorithm[:c_idx]
                                final_algorithm = ','.join(new_commands_algorithm)

                                return final_algorithm, c_idx
                            else:
                                raise ValueError('Invalid direction: ', direction)

                    new_coordinates = ','.join(new_coordinates)

                    if len(new_coordinates) == 2:
                        final_algorithm = ','.join([f'go({new_coordinates}),paint({color})'])
                    else:
                        final_algorithm = ','.join([f'paint({{{color}}},{{{new_coordinates}}})'])

                return final_algorithm, c_idx
            else:
                # There is an initial part of the algorithm that is missing, so insert it
                paint_command_idx = commands_algorithm.index(paint_command)
                initial_part_final_algorithm = ','.join(commands_algorithm[:paint_command_idx - 1])

                new_commands_algorithm = commands_algorithm[paint_command_idx - 1:]
                #  Then behave like the normal case of paint_command_pattern_match
                if len(new_commands_algorithm) > 1:
                    if new_commands_algorithm[0].startswith('go(') and new_commands_algorithm[1] == paint_command:
                        second_part_final_algorithm = new_commands_algorithm[0] + ',' + paint_command
                        updated_retrieved_commands = [new_commands_algorithm[0], paint_command]

                        matching_indices = []

                        for i in range(len(new_commands_algorithm) - len(updated_retrieved_commands) + 1):
                            if new_commands_algorithm[i:i + len(updated_retrieved_commands)] == updated_retrieved_commands:
                                matching_indices.append(i)

                        if len(matching_indices) == 0:
                            raise ValueError('No matching index found')

                        if matching_indices[0] != 0:
                            raise ValueError('Invalid matching index: ', matching_indices[0])

                        if len(matching_indices) >= 1:
                            c_idx = matching_indices[0] + 2 + (paint_command_idx - 1)
                            final_algorithm = initial_part_final_algorithm + ',' + second_part_final_algorithm
                            return final_algorithm, c_idx
                        else:
                            pass
                    else:
                        if new_commands_algorithm[0].startswith('go(') and paint_command in new_commands_algorithm:
                            missing_commands = new_commands_algorithm[:new_commands_algorithm.index(paint_command)]
                            missing_paint_commands = [element for element in missing_commands if
                                                      not re.match(regex_go, element)]
                            if all(element == missing_paint_commands[0] for element in missing_paint_commands):
                                beginning_final_algorithm, idx1 = get_paint_pattern(new_commands_algorithm,
                                                                                    [missing_paint_commands[0]],
                                                                                    regex_go, regex_dot, regex_pattern)
                                ending_final_algorithm, idx2 = get_paint_pattern(new_commands_algorithm[idx1:],
                                                                                 [paint_command],
                                                                                 regex_go, regex_dot, regex_pattern)
                                second_part_final_algorithm = beginning_final_algorithm + ',' + ending_final_algorithm

                                # c_idx updated
                                c_idx = idx1 + idx2 + (paint_command_idx - 1)
                                final_algorithm = initial_part_final_algorithm + ',' + second_part_final_algorithm

                                return final_algorithm, c_idx
                            else:
                                pass
                        else:
                            raise ValueError('Invalid commands: ', commands_algorithm)

        elif paint_command_dot_match:
            coordinates = ''
            while c_idx < len(commands_algorithm):
                command = commands_algorithm[c_idx]
                command_dot_match = re.match(regex_dot, command)
                if command.startswith('go('):
                    if next_sublist_command is not None and command == next_sublist_command:
                        break
                    no_next_command = len(commands_algorithm) == c_idx + 1
                    if no_next_command:
                        break
                    next_command = commands_algorithm[c_idx + 1]
                    if paint_command in next_command:
                        go_parameter = command.split('(')[1].split(')')[0]

                        coordinates += go_parameter + ','
                        c_idx += 1
                    else:
                        if not coordinates:
                            missing_algorithm += command + ',' + next_command
                            c_idx += 2
                        else:
                            # c_idx don't need to be updated
                            break
                elif command_dot_match:
                    c_idx += 1
                elif command.startswith('fill_empty('):
                    break
                elif command.startswith('copy('):
                    break
                elif command.startswith('mirror('):
                    break
                else:
                    raise ValueError('Invalid command: ', command)

            # Remove last comma from coordinates
            if next_sublist_command is not None and not coordinates:
                return "", c_idx
            elif not coordinates:
                raise ValueError('Invalid coordinates: ', coordinates)
            else:
                if coordinates[-1] == ',':
                    coordinates = coordinates[:-1]
                else:
                    raise ValueError('Invalid end for coordinates: ', coordinates)

            # Append coordinates to the list
            go_coordinates_list.append(coordinates)

            # Create the final algorithm from the coordinates and colours
            for go_coordinates, paint_colours in zip(go_coordinates_list, paint_colours_list):
                if len(go_coordinates) == 2:
                    final_algorithm = ','.join([f'go({go_coordinates}),paint({paint_colours})'])
                else:
                    final_algorithm = ','.join([f'paint({{{paint_colours}}},{{{go_coordinates}}})'])

            if missing_algorithm:
                final_algorithm = missing_algorithm + ',' + final_algorithm

            return final_algorithm, c_idx
        elif paint_command_pattern_match:
            if len(commands_algorithm) > 1:
                if commands_algorithm[0].startswith('go(') and commands_algorithm[1] == paint_command:
                    final_algorithm = commands_algorithm[0] + ',' + paint_command
                    updated_retrieved_commands = [commands_algorithm[0], paint_command]

                    matching_indices = []

                    for i in range(len(commands_algorithm) - len(updated_retrieved_commands) + 1):
                        if commands_algorithm[i:i + len(updated_retrieved_commands)] == updated_retrieved_commands:
                            matching_indices.append(i)

                    if len(matching_indices) == 0:
                        raise ValueError('No matching index found')

                    if matching_indices[0] != 0:
                        raise ValueError('Invalid matching index: ', matching_indices[0])

                    if len(matching_indices) >= 1:
                        c_idx = matching_indices[0] + 2
                        return final_algorithm, c_idx
                    else:
                        pass  # TODO
                else:
                    if commands_algorithm[0].startswith('go(') and paint_command in commands_algorithm:
                        missing_commands = commands_algorithm[:commands_algorithm.index(paint_command)]
                        missing_paint_commands = [element for element in missing_commands if
                                                  not re.match(regex_go, element)]
                        if all(element == missing_paint_commands[0] for element in missing_paint_commands):
                            beginning_final_algorithm, idx1 = get_paint_pattern(commands_algorithm,
                                                                                [missing_paint_commands[0]],
                                                                                regex_go, regex_dot, regex_pattern)
                            ending_final_algorithm, idx2 = get_paint_pattern(commands_algorithm[idx1:], [paint_command],
                                                                             regex_go, regex_dot, regex_pattern)
                            final_algorithm = beginning_final_algorithm + ',' + ending_final_algorithm

                            c_idx = idx1 + idx2
                            return final_algorithm, c_idx
                        else:
                            pass  # TODO
                    else:
                        raise ValueError('Invalid commands: ', commands_algorithm)
            else:
                if len(commands_algorithm) == 1 and len(commands_retrieved_algorithm) == 1:
                    if commands_algorithm[0].startswith('go(') and \
                            not commands_retrieved_algorithm[0].startswith('go('):
                        final_algorithm = commands_algorithm[0] + ',' + commands_retrieved_algorithm[0]
                        c_idx = 2
                        return final_algorithm, c_idx
                    else:
                        # The coordinate is missing in the algorithm so return the commands_algorithm as it is
                        if commands_algorithm == commands_retrieved_algorithm:
                            c_idx += len(commands_algorithm)
                            final_algorithm = ','.join(commands_algorithm)
                            return final_algorithm, c_idx
                        else:
                            pass  # TODO
                else:
                    return "", c_idx
        else:
            continue


def get_matching_pattern(input_string, regex_go, regex_dot, regex_pattern, regex_fillempty, regex_copy, regex_mirror):
    """
    Get the matching pattern for the input string
    :param input_string:    the input string
    :param regex_go:        the regular expression for the go command
    :param regex_dot:       the regular expression for the dot command
    :param regex_pattern:   the regular expression for the pattern command
    :param regex_fillempty: the regular expression for the fillempty command
    :param regex_copy:      the regular expression for the copy command
    :param regex_mirror:    the regular expression for the mirror command
    :return:                the matching pattern
    """
    # Check which regex pattern the input string matches
    matching_pattern = None

    regex_coordinate = r'[A-F][1-6]'

    if re.match(regex_go, input_string):
        matching_pattern = 'go'
    elif re.match(regex_dot, input_string) or re.match(regex_pattern, input_string):
        matching_pattern = 'paint'
    elif re.match(regex_fillempty, input_string):
        matching_pattern = 'fillempty'
    elif re.match(regex_copy, input_string):
        matching_pattern = 'copy'
    elif re.match(regex_mirror, input_string):
        matching_pattern = 'mirror'
    elif re.match(regex_coordinate, input_string):
        matching_pattern = 'go'

    return matching_pattern


def filter_and_track_indexes(paint_commands_list):
    """
    Filter the consecutive paint commands and track the indices of the paint commands
    :param paint_commands_list: the list of paint commands
    :return:                    the filtered list of paint commands and the indices of the paint commands
    """
    filtered_list = []
    indexes = []
    for i, command in enumerate(paint_commands_list):
        if i == 0 or command != paint_commands_list[i - 1]:
            filtered_list.append(command)
            indexes.append(i)

    return filtered_list, indexes


def keep_elements_from_index(original_list, indices_to_keep):
    """
    Keep the elements in the original list corresponding to the indices to keep
    :param original_list:   the original list
    :param indices_to_keep: the list of indices to keep
    :return:                the new list
    """
    new_list = []
    for index, element in enumerate(original_list):
        if index in indices_to_keep:
            new_list.append(element)

    return new_list


def remove_elements_from_index(original_list, list2_indices, indices_to_remove, elements_to_remove, new_list=None):
    """
    Remove the elements in the original list corresponding to the indices to remove
    :param original_list:      the original list
    :param list2_indices:      the indices of the elements in the original list
    :param indices_to_remove:  the list of indices to remove
    :param elements_to_remove: the elements to remove from list1 to list2
    :param new_list:           the new list, initially empty
    :return:                   the new list
    """
    if new_list is None:
        new_list = []

    actual_indices_to_remove = []
    for i, _ in enumerate(indices_to_remove):
        index = list2_indices[indices_to_remove[i]]
        actual_indices_to_remove.append(index)

    if [original_list[i] for i in actual_indices_to_remove] != elements_to_remove:
        raise ValueError('Invalid elements to remove: ', [original_list[i] for i in actual_indices_to_remove],
                         ' and elements to remove: ', elements_to_remove)

    for e, element in enumerate(original_list):
        if e not in indices_to_remove:
            new_list.append(element)

    return new_list


def add_elements_from_index(original_list2, indices_to_add, elements_to_add):
    """
    Add the elements in the original list corresponding to the indices to add
    :param original_list2:      the original list2
    :param indices_to_add:      the list of indices to add
    :param elements_to_add:     the element to add from list1 to list2
    :return:                    the new list2
    """
    new_list2 = original_list2.copy()

    for idx, element in enumerate(elements_to_add):
        if len(elements_to_add) > len(indices_to_add):
            raise ValueError('Invalid length of elements_to_add: ', len(elements_to_add), ' and indices_to_add: ',
                             len(indices_to_add))
        index_to_add = indices_to_add[idx]

        new_list2.insert(index_to_add, element)

    return new_list2


def get_commands_list(algorithm):
    """
    Get the commands list from the algorithm
    :param algorithm: the algorithm
    :return:          the commands list
    """
    commands = []
    command = ''
    open_brackets = 0

    for char in algorithm:
        command += char
        if char in ['(', '{']:
            open_brackets += 1
        elif char in [')', '}']:
            open_brackets -= 1

        if char == ',' and open_brackets == 0:
            commands.append(command[:-1].strip())
            command = ''

    if command:
        commands.append(command.strip())

    return commands


def remove_initial_comma_in_algorithm(algorithm):
    """
    Remove the initial comma in the algorithm string recursively
    :param algorithm: the algorithm string
    :return:          the algorithm string without the initial comma
    """
    if algorithm.startswith(','):
        algorithm = algorithm[1:]
        return remove_initial_comma_in_algorithm(algorithm)
    else:
        return algorithm


def remove_final_comma_in_algorithm(algorithm):
    """
    Remove the final comma in the algorithm string recursively
    :param algorithm: the algorithm string
    :return:          the algorithm string without the final comma
    """
    if algorithm.endswith(','):
        algorithm = algorithm[:-1]
        return remove_final_comma_in_algorithm(algorithm)
    else:
        return algorithm


def remove_multiple_commas_in_algorithm(algorithm):
    """
    Remove multiple commas in the algorithm string
    :param algorithm: the algorithm string
    :return:          the algorithm string without multiple commas
    """
    if bool(re.search(r',,', algorithm)):
        new_algorithm = algorithm.replace(r',,', r',')

        return new_algorithm
    else:
        return algorithm
