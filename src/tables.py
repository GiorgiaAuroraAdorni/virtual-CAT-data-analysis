"""
@file-name:     tables.py
@date-creation: 02.09.2023
@author-name:   Giorgia Adorni
"""
import numpy as np
import pandas as pd
from tabulate import tabulate

from utils import save_and_print_table, format_time


def performance_by_schema_and_age_latex_table(dataframe, caption, label, tables_output_folder):
    """
    Print a table with performance data grouped by schema and age category.
    :param dataframe: DataFrame with the results.
    :param caption: Caption for the table.
    :param label: Label for the table.
    :param tables_output_folder: Output folder to save the table files.
    :return: None
    """
    dimension_name = 'AGE_GROUP'
    category_name = 'AGE_CATEGORY'

    age_category_order = dataframe.groupby(dimension_name)[category_name].unique().explode().unique().tolist()

    # Convert SCHEMA_ID and category to categorical data types with the specified order
    dataframe[category_name] = pd.Categorical(dataframe[category_name], categories=age_category_order, ordered=True)

    # Ensure AGE_CATEGORY is treated as a categorical variable
    dataframe['AGE_CATEGORY'] = dataframe['AGE_CATEGORY'].astype('category')

    # Loop through each AGE_CATEGORY and generate a table
    for age_category in dataframe['AGE_CATEGORY'].cat.categories:
        age_df = dataframe[dataframe['AGE_CATEGORY'] == age_category]

        # Call the existing function to generate a table for the current age category
        performance_by_schema_latex_table(age_df, f'{caption} - Age Category: {age_category}',
                                          f'{label}_age_{age_category}', tables_output_folder, age_category)


def performance_by_schema_latex_table(dataframe, caption, label, tables_output_folder, age_category=None):
    """
    Print a table with the number of pupils who solved the schema,
    the number of pupils who correctly solved the schema,
    the number of different algorithms, the percentage of algorithms with 0D, 1D and 2D
    :param dataframe:            DataFrame with the results
    :param caption:              Caption for the table
    :param label:                Label for the table
    :param tables_output_folder: Output folder to save the table files
    :param age_category:         Age category to filter the DataFrame
    """
    # Group by SCHEMA_ID and calculate counts and percentages
    grouped = dataframe.groupby('SCHEMA_ID').agg({
        'STUDENT_ID': 'count',
        'COMPLETE': 'sum',
        'CORRECT': 'sum'
    })

    total_students = grouped['STUDENT_ID'].max()

    # Calculate percentages and format columns
    grouped['COMPLETE_percentage'] = (grouped['COMPLETE'] / total_students * 100).apply(lambda x: f'{x:.0f}')
    grouped['CORRECT_percentage'] = (grouped['CORRECT'] / grouped['COMPLETE'] * 100).apply(lambda x: f'{x:.0f}')

    # Create a new DataFrame with formatted columns
    new_df = pd.DataFrame({
        'SCHEMA_ID': grouped.index,
        'Num. pupils who completed the schema': grouped['COMPLETE'].astype(int).astype(str) +
                                                '/' + total_students.astype(int).astype(str) +
                                                ' (' + grouped['COMPLETE_percentage'] + '\%)',
        'Num. pupils who solved the schema': grouped['CORRECT'].astype(int).astype(str) +
                                             '/' + grouped['COMPLETE'].astype(int).astype(str) +  # Use 'COMPLETE' here
                                             ' (' + grouped['CORRECT_percentage'] + '\%)',
        # Use 'CORRECT_percentage' here
    })

    # Reset the index of the new DataFrame
    new_df = new_df.reset_index(drop=True)

    # Specify the first row as the header with bold formatting
    header_row = [r'\textbf{Schema}',
                  r'\multicolumn{1}{>{\centering\arraybackslash}m{2.2cm}}{\textbf{Num. pupils who attempted the schema}}',
                  r'\multicolumn{1}{>{\centering\arraybackslash}m{2.2cm}}{\textbf{Num. pupils who solved the schema}}']

    # Add the column widths to the LaTeX code
    ws = r'c>{\raggedleft\arraybackslash}m{2.2cm}>{\raggedleft\arraybackslash}m{2.2cm}'

    # Generate LaTeX code with custom column widths
    latex_code = tabulate(new_df, headers=header_row, tablefmt='latex_raw', colalign=['c'] * len(header_row),
                          showindex=False)

    latex_code = latex_code.replace(r'{lll}', r'{' + ''.join(ws) + '}')

    if age_category is not None:
        save_and_print_table(latex_code, caption, label, tables_output_folder, 'performance_by_schema_{}.tex'.format(age_category))
    else:
        save_and_print_table(latex_code, caption, label, tables_output_folder, 'performance_by_schema.tex')


def performance_by_category_latex_table(dataframe, category, caption, label, tables_output_folder, filename):
    """
    Generate a LaTeX table with the schema by student analysis
    :param dataframe:            The DataFrame containing the data
    :param category:             The category to filter the DataFrame
    :param caption:              The caption for the table
    :param label:                The label for the table
    :param tables_output_folder: Output folder to save the table files
    :param filename:             The filename for the table
    """
    if category == 'ARTEFACT':
        first_column_header = 'Interaction dimension'
        dimension_name = category + '_DIMENSION'
        category_name = category + '_TYPE'
    else:
        first_column_header = 'Age category'
        dimension_name = category + '_GROUP'
        category_name = category + '_CATEGORY'

    category_order = dataframe.groupby(dimension_name)[category_name].unique().explode().unique().tolist()

    # Convert SCHEMA_ID and category to categorical data types with the specified order
    dataframe[category_name] = pd.Categorical(dataframe[category_name],
                                              categories=category_order,
                                              ordered=True)

    # Remove useless columns
    dataframe = dataframe[['STUDENT_ID', 'SCHEMA_ID', 'COMPLETE', 'CORRECT', category_name]]

    student_dataframe = dataframe[['STUDENT_ID', 'COMPLETE', 'CORRECT', category_name]]

    student_number = student_dataframe['STUDENT_ID'].nunique()

    # Create a new dataframe with a row for each student
    category_dataframe = dataframe[['STUDENT_ID', category_name]].drop_duplicates()

    if category == 'AGE':
        # Add to the dataframe the column complete
        category_dataframe.loc[:, 'COMPLETE'] = student_dataframe.apply(
            lambda row: all(dataframe[dataframe['STUDENT_ID'] == row['STUDENT_ID']]['COMPLETE']), axis=1)

        # Add to the dataframe the column correct
        category_dataframe.loc[:, 'CORRECT'] = student_dataframe.apply(
            lambda row: all(dataframe[dataframe['STUDENT_ID'] == row['STUDENT_ID']]['CORRECT']), axis=1)

    else:
        # Add to the dataframe the column complete
        category_dataframe.loc[:, 'COMPLETE'] = category_dataframe.apply(
            lambda row: any(dataframe[(dataframe['STUDENT_ID'] == row['STUDENT_ID']) & (
                        dataframe[category_name] == row[category_name])]['COMPLETE']), axis=1)

        # Add to the dataframe the column correct
        category_dataframe.loc[:, 'CORRECT'] = category_dataframe.apply(
            lambda row: any(dataframe[(dataframe['STUDENT_ID'] == row['STUDENT_ID']) & (
                        dataframe[category_name] == row[category_name])]['CORRECT']), axis=1)

    # Group by the category and calculate the number of students who attempted all schemas
    grouped = category_dataframe.groupby(category_name, observed=False).agg({
        'STUDENT_ID': 'count',
        'COMPLETE': 'sum',
        'CORRECT': 'sum'
    }).reset_index()

    # Format columns
    if category == 'AGE':
        grouped['column_COMPLETE'] = (grouped['COMPLETE'].astype(int).astype(str) +
                                      '/' + grouped['STUDENT_ID'].astype(int).astype(str) +
                                      ' (' + (grouped['COMPLETE'].astype(int) / grouped['STUDENT_ID'].astype(int) * 100).round().astype(int).astype(str) + '\%)')
        grouped['column_CORRECT'] = (grouped['CORRECT'].astype(int).astype(str) +
                                     '/' + grouped['COMPLETE'].astype(int).astype(str) +
                                     ' (' + (grouped['CORRECT'].astype(int) / grouped['COMPLETE'].astype(int) * 100).round().astype(int).astype(str) + '\%)')
    else:
        grouped['column_COMPLETE'] = (grouped['COMPLETE'].astype(int).astype(str) +
                                      '/' + str(student_number) +
                                      ' (' + (grouped['COMPLETE'].astype(int) / student_number * 100).round().astype(int).astype(str) + '\%)')
        grouped['column_CORRECT'] = (grouped['CORRECT'].astype(int).astype(str) +
                                     '/' + grouped['COMPLETE'].astype(int).astype(str) +
                                     ' (' + (grouped['CORRECT'].astype(int) / grouped['COMPLETE'].astype(int) * 100).round().astype(int).astype(str) + '\%)')

    # Create the new DataFrame with the specified columns
    full_df = pd.DataFrame({
        first_column_header: grouped[category_name],
        'Complete': grouped['column_COMPLETE'],
        'Correct': grouped['column_CORRECT']
    })

    # Calculate the number of schemas solved by each student in each category
    student_schemas_solved = dataframe.groupby(['STUDENT_ID', category_name], observed=False)[
        ['COMPLETE', 'CORRECT']].sum().reset_index()

    # Create a list of unique pairs of STUDENT_ID and category
    unique_pairs = dataframe[['STUDENT_ID', category_name]].drop_duplicates()

    # Filter the DataFrame to keep only the rows with the unique pairs
    filtered_student_schemas_solved = student_schemas_solved.merge(unique_pairs, on=['STUDENT_ID', category_name])

    # Make the columns COMPLETE and CORRECT int type
    filtered_student_schemas_solved['COMPLETE'] = filtered_student_schemas_solved['COMPLETE'].astype(int)
    filtered_student_schemas_solved['CORRECT'] = filtered_student_schemas_solved['CORRECT'].astype(int)

    # Calculate the average, standard deviation, median, quantile and mode of schemas solved for each category
    avg_std_schemas_solved = filtered_student_schemas_solved.groupby(category_name, observed=False)[
        ['COMPLETE', 'CORRECT']].agg(
        ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]).reset_index()
    avg_std_schemas_solved.columns = [category_name, 'Median Complete', 'Q1 Complete', 'Q3 Complete',
                                      'Median Correct', 'Q1 Correct', 'Q3 Correct']

    # Add the MEDIAN_COMPLETE and MEDIAN_CORRECT columns with median (Q1, Q3)
    full_df['MEDIAN_COMPLETE'] = avg_std_schemas_solved['Median Complete'].apply(lambda x: f'{x:.0f}') + ' (' \
                                 + avg_std_schemas_solved['Q1 Complete'].apply(lambda x: f'{x:.0f}') + '-' \
                                 + avg_std_schemas_solved['Q3 Complete'].apply(lambda x: f'{x:.0f}') + ')'
    full_df['MEDIAN_CORRECT'] = avg_std_schemas_solved['Median Correct'].apply(lambda x: f'{x:.0f}') + ' (' \
                                + avg_std_schemas_solved['Q1 Correct'].apply(lambda x: f'{x:.0f}') + '-' \
                                + avg_std_schemas_solved['Q3 Correct'].apply(lambda x: f'{x:.0f}') + ')'

    # Calculate the complete count and percentage
    complete_count = dataframe.groupby('STUDENT_ID')['COMPLETE'].all().sum()
    complete_percentage = round((complete_count / dataframe['STUDENT_ID'].nunique()) * 100)

    # Calculate the correct count and percentage
    student_total = dataframe['STUDENT_ID'].nunique()
    correct_count = dataframe.groupby('STUDENT_ID')['CORRECT'].all().sum()
    correct_percentage = round((correct_count / dataframe['STUDENT_ID'].nunique()) * 100)

    # Calculate the average, standard deviation, median, iqr and mode of completed schemas per student
    median_completed = dataframe.groupby('STUDENT_ID')['COMPLETE'].sum().median()
    iqr_completed = dataframe.groupby('STUDENT_ID')['COMPLETE'].sum().quantile([0.25, 0.75])

    # Calculate the average, standard deviation, median, iqr and mode of correct schemas per student
    median_correct = dataframe.groupby('STUDENT_ID')['CORRECT'].sum().median()
    iqr_correct = dataframe.groupby('STUDENT_ID')['CORRECT'].sum().quantile([0.25, 0.75])

    # Create the table data
    table_data = [[
        r'\textbf{Total}',
        f'{complete_count}/{student_total} ({complete_percentage}\%)',
        f'{correct_count}/{complete_count} ({correct_percentage}\%)',
        f'{median_completed:.0f} ({iqr_completed[0.25]:.0f}-{iqr_completed[0.75]:.0f})',
        f'{median_correct:.0f} ({iqr_correct[0.25]:.0f}-{iqr_correct[0.75]:.0f})'
    ]]

    # Convert the list to a DataFrame
    new_row_df = pd.DataFrame(table_data, columns=full_df.columns)

    # Append the new row to the original DataFrame
    full_df = pd.concat([full_df, new_row_df], ignore_index=True)

    print(full_df)

    # Define the header row
    header_row = [r'\textbf{{{}}}'.format(first_column_header),
                  r'\multicolumn{1}{>{\centering\arraybackslash}m{2cm}}{\textbf{Num. of pupils who attempted all schemas}}',
                  r'\multicolumn{1}{>{\centering\arraybackslash}m{2cm}}{\textbf{Num. of pupils who solved all schemas}}',
                  r'{\textbf{Num. of \linebreak attempted schemas}}',
                  r'{\textbf{Num. of \linebreak solved schemas}}']

    # Generate LaTeX code with custom column widths
    latex_code = tabulate(full_df, headers=header_row, tablefmt='latex_raw', colalign=['c'] * len(header_row),
                          showindex=False)

    # Add the column widths to the LaTeX code
    ws = r'l>{\raggedleft\arraybackslash}m{2cm}>{\raggedleft\arraybackslash}m{2cm}' \
         r'>{\centering\arraybackslash}m{2cm}>{\centering\arraybackslash}m{2cm}'

    latex_code = latex_code.replace(r'{lllll}', r'{' + ''.join(ws) + '}')

    # Insert the \cmidrule command before the last row
    midrule = r'\cmidrule(lr){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}'
    latex_code = latex_code.replace(r'\textbf{Total}', midrule + r'\textbf{Total}')

    save_and_print_table(latex_code, caption, label, tables_output_folder, filename)


def algorithms_latex_table(dataframe, caption, label, tables_output_folder):
    """
    Generate a LaTeX table with algorithm analysis data
    :param dataframe: Pandas DataFrame
    :param caption:   Caption of the table
    :param label:     Label of the table
    :return:          None
    """
    # Get unique SCHEMA_ID values
    unique_schema_ids = dataframe['SCHEMA_ID'].unique()

    # Create an empty dictionary to store frequency data
    frequency_data = {}

    # Iterate over unique SCHEMA_ID values
    for schema_id in unique_schema_ids:
        # Filter the DataFrame for the current SCHEMA_ID
        schema_data = dataframe[dataframe['SCHEMA_ID'] == schema_id]

        # Count the frequency of each ALGORITHM_ID
        algorithm_freq = schema_data['ALGORITHM_ID'].value_counts().reset_index()['count']

        # Calculate the total count of ALGORITHM_ID
        total_count = algorithm_freq.sum()

        # Calculate the percentages and convert to strings with the % symbol
        algorithm_freq = (algorithm_freq / total_count * 100).round(2).astype(str) + '\%'

        # Store the percentages in the dictionary using SCHEMA_ID as the key
        frequency_data[schema_id] = algorithm_freq

    # Create the frequency DataFrame using the dictionary
    frequency_df = pd.DataFrame(frequency_data)

    # Fill NaN values with empty strings
    frequency_df = frequency_df.fillna('')

    # Change the index to range(1, 30)
    frequency_df = frequency_df.set_index(pd.Index(range(1, len(frequency_df) + 1)))

    header_row = []
    for cell in unique_schema_ids:
        b = r'\multicolumn{1}{c}{'
        header_row.append(b + r'\textbf{{{}}}'.format(cell) + r'}')

    # Generate LaTeX code for the table
    latex_code = tabulate(frequency_df, tablefmt='latex_raw', headers=header_row)

    # Add the necessary LaTeX code to make the first column bold
    latex_code = latex_code.replace(r'{rllllllllllll}', r'{>{\bfseries}c|rrrrrrrrrrrr}')

    latex_code = latex_code.replace(r'& \multicolumn{1}{c}{\textbf{1}}', r'& \multicolumn{12}{c}{\textbf{Schemas}}\\ & \textbf{1}')

    save_and_print_table(latex_code, caption, label, tables_output_folder, 'algorithms.tex')


def students_latex_table(dataframe, caption, label, tables_output_folder):
    """
    Generate a LaTeX table with student analysis data
    :param dataframe:            Pandas DataFrame
    :param caption:              Caption of the table
    :param label:                Label of the table
    :param tables_output_folder: Output folder to save the table files
    """
    # Get the student data
    students_df = dataframe.groupby('STUDENT_ID')[
        ['AGE', 'AGE_GROUP', 'AGE_CATEGORY', 'GENDER', 'SCHOOL_TYPE']].first().reset_index()

    # Get the age categories and school types
    age_categories = students_df.groupby('AGE_GROUP')['AGE_CATEGORY'].unique().explode().unique().tolist()
    school_types = students_df.groupby('AGE_GROUP')['SCHOOL_TYPE'].unique().explode().unique().tolist()

    if len(school_types) > 2:
        school_types = ['Preschool', 'Primary', 'Primary - Secondary', 'Secondary']
    else:
        school_types = ['Preschool', 'Primary', 'Secondary']

    # Compute the number of students with GENDER equal to Female, Male, and the total number of students
    female_counts = students_df[students_df['GENDER'] == 'Female'].groupby('AGE_GROUP').size().tolist()
    male_counts = students_df[students_df['GENDER'] == 'Male'].groupby('AGE_GROUP').size().tolist()
    total_counts = students_df.groupby('AGE_GROUP').size().tolist()

    # Iterate over the age_categories list
    updated_age_categories = []
    for i, category in enumerate(age_categories):
        # Extract the age range from the original string
        age_range = category.split(' ')[1:4:2]
        what = category.split(' ')[4:]

        # Transform the age range into the desired format
        transformed_string = '–'.join(age_range) + ' ' + ' '.join(what)

        # Filter the DataFrame for the current category
        category_data = students_df[students_df['AGE_CATEGORY'] == category]

        # Calculate the mean and standard deviation of the age
        mean_age = category_data['AGE'].mean()
        std_age = category_data['AGE'].std()

        # Append the mean and standard deviation to the current category element in the list
        updated_age_categories.append(transformed_string + ' ($\mu$ = {:.1f} $\pm$ {:.1f})'.format(mean_age, std_age))

    # Select only the necessary columns for the table
    table_data = [school_types, updated_age_categories, female_counts, male_counts, total_counts]
    table_data = [[element for element in tpl] for tpl in table_data]

    # Compute the total number of students for gender and append it to the table_data list
    totals_row = [r'\textbf{{{Total}}}', ''] + [sum(t) for t in table_data[-3:]]
    table_data = [element + [totals_row[i]] for i, element in enumerate(table_data)]

    table_data = list(zip(*table_data))

    # Insert the \cmidrule command before the last row
    midrule = r'\cmidrule(lr){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}'
    # table_data.insert(-1, [midrule])

    header_row = []
    for cell in ['School type', 'Age category', 'Female', 'Male', 'Total']:
        header_row.append(r'\textbf{{{}}}'.format(cell))

    # Generate LaTeX code for the table
    latex_code = tabulate(table_data, headers=header_row, tablefmt='latex_raw')

    latex_code = latex_code.replace(r'{llrrr}', r'{llccc}')
    latex_code = latex_code.replace(r'\textbf{{{Total}}}', midrule + r'\textbf{{{Total}}}')

    save_and_print_table(latex_code, caption, label, tables_output_folder, 'students.tex')


def students_by_school_latex_table(dataframe, caption, label, tables_output_folder):
    """
    Generate a LaTeX table with student analysis data
    :param dataframe:            Pandas DataFrame
    :param caption:              Caption of the table
    :param label:                Label of the table
    :param tables_output_folder: Output folder to save the table files
    """
    # Get the student data
    students_df = dataframe.groupby('STUDENT_ID')[
        ['AGE', 'AGE_GROUP', 'AGE_CATEGORY', 'HARMOS_GRADE', 'SESSION_ID', 'GENDER',
         'SCHOOL_ID', 'SCHOOL_TYPE', 'CANTON_NAME']].first().reset_index()

    # Get the age categories and school types
    sessions = students_df.groupby('SESSION_ID')['SESSION_ID'].unique().explode().unique().tolist()
    age_categories = students_df.groupby(['SESSION_ID'])['AGE_CATEGORY'].unique().explode().values.tolist()
    harmos = students_df.groupby(['SESSION_ID', 'SCHOOL_TYPE'])['HARMOS_GRADE'].unique().values.tolist()
    harmos_strings = []
    # for each np.array in harmos, sort it and convert it to a string
    for array in harmos:
        array.sort()
        harmos_strings.append(', '.join(array.astype(str)))
    school_ids = students_df.groupby('SESSION_ID')['SCHOOL_ID'].unique().explode().values.tolist()
    school_types = students_df.groupby('SESSION_ID')['SCHOOL_TYPE'].unique().explode().values.tolist()
    cantons = students_df.groupby('SESSION_ID')['CANTON_NAME'].unique().explode().values.tolist()

    # Create a new variable schools that is a list of strings with the school_id and school_type
    schools = [f'{school_id} \quad {school_type}' for school_id, school_type in zip(school_ids, school_types)]

    # Rename cantons using a mapping
    canton_mapping = {
        'Ticino (TI)': 'Ticino',
        'Solothurn (SO)': 'Solothurn'}
    cantons = [canton_mapping.get(item, item) for item in cantons]

    # Raname age categories using a mapping
    age_category_mapping = {
        'From 3 to 6 years old':   '3-6 & yrs',
        'From 7 to 9 years old':   '7-9 & yrs',
        'From 10 to 13 years old': '10-13 & yrs',
        'From 14 to 16 years old': '14-16 & yrs'}
    updated_age_categories = [age_category_mapping.get(item, item) for item in age_categories]

    # Compute the number of students with GENDER equal to Female, Male, and the total number of students,
    # for each session in sessions, if there are no students of a certain Gender, the count is 0
    female_counts = [students_df[students_df['GENDER'] == 'Female'][students_df['SESSION_ID'] == session].shape[0] for session in sessions]
    male_counts = [students_df[students_df['GENDER'] == 'Male'][students_df['SESSION_ID'] == session].shape[0] for session in sessions]
    total_counts = [students_df[students_df['SESSION_ID'] == session].shape[0] for session in sessions]

    # Compute mean and std of age for each session
    for i, session in enumerate(sessions):
        # Filter the DataFrame for the current category
        session_data = students_df[students_df['SESSION_ID'] == session]

        # Calculate the mean and standard deviation of the age
        mean_age = session_data['AGE'].mean()
        std_age = session_data['AGE'].std()

        # Append the mean and standard deviation to the current category element in the list
        updated_age_categories[i] = updated_age_categories[i] + '& ($\mu$ {:.1f} $\pm$ {:.1f} yrs)'.format(mean_age, std_age)

    # Select only the necessary columns for the table
    table_data = [sessions, cantons, schools, harmos_strings, updated_age_categories, female_counts, male_counts, total_counts]
    table_data = [[element for element in tpl] for tpl in table_data]

    # Compute the average and std age of the entire dataset
    mean_age = students_df['AGE'].mean()
    std_age = students_df['AGE'].std()
    string_age = '&&($\mu$ {:.1f} $\pm$ {:.1f} yrs)'.format(mean_age, std_age)

    # Compute the total number of students for gender and append it to the table_data list
    totals_row = [r'\textbf{{{Total}}}', '', '', '', string_age] + [sum(t) for t in table_data[-3:]]
    table_data = [element + [totals_row[i]] for i, element in enumerate(table_data)]

    table_data = list(zip(*table_data))

    # Insert the \cmidrule command before the last row
    midrule = r'\cmidrule(lr){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}\cmidrule(lr){9-9}\cmidrule(lr){10-10}'
    # table_data.insert(-1, [midrule])

    header_row = []
    for cell in ['Session', 'Canton', 'School ID \& type', 'HarmoS', 'Age category', 'Female', 'Male', 'Total']:
        header_row.append(r'\textbf{{{}}}'.format(cell))

    # Modify the header row at index 3, the one for age category adding at the beginning \multicolumn{2}{l}{ and at the end }
    header_row[4] = r'\multicolumn{3}{c}{' + header_row[4] + '}'

    # Generate LaTeX code for the table
    latex_code = tabulate(table_data, headers=header_row, tablefmt='latex_raw')

    latex_code = latex_code.replace(r'{lllrlrrr}', r'{cllcc@{\hspace{.5\tabcolsep}}c@{\hspace{.8\tabcolsep}}cccc}')
    latex_code = latex_code.replace(r'\textbf{{{Total}}}', midrule + r'\textbf{{{Total}}}')

    save_and_print_table(latex_code, caption, label, tables_output_folder, 'students_by_school.tex')


def time_by_schema_latex_table(dataframe, caption, label, tables_output_folder, std=False):
    """
    Generate a LaTeX table with time analysis data by schema
    :param dataframe:            Pandas DataFrame
    :param caption:              Caption of the table
    :param label:                Label of the table
    :param tables_output_folder: Output folder to save the table files
    :param std:                  Boolean indicating whether to include the standard deviation or not in the final table
    """
    statistics = ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    header_columns = ['Schema', 'Median', 'Q1', 'Q3']
    columns = ['Median', 'Q1', 'Q3']
    table_orginal_format = r'{rlll}'
    table_new_format = r'{cccc}'

    # Group by SCHEMA_ID and calculate the statistics
    grouped_data = dataframe.groupby(['SCHEMA_ID'])['LOG_TIME'].agg(statistics).reset_index()

    # If the statistics contain a time equal to 0, make it equal to 1
    grouped_data = grouped_data.apply(lambda col: col.map(lambda x: 1 if x == 0 else x))

    # Rename the last two columns of grouped_data using the last two elements from the list columns
    grouped_data = grouped_data.rename(columns=dict(zip(grouped_data.columns[-3:], columns)))

    # Then convert values from seconds to minutes and seconds and represent as strings
    grouped_data[columns] = grouped_data[columns].apply(lambda col: col.map(format_time))

    # Modify column names
    header_row = []
    for cell in header_columns:
        header_row.append(r'\textbf{{{}}}'.format(cell))

    # Generate LaTeX code for the table
    latex_code = tabulate(grouped_data, headers=header_row, tablefmt='latex_raw', showindex=False)
    latex_code = latex_code.replace(table_orginal_format, table_new_format)

    save_and_print_table(latex_code, caption, label, tables_output_folder, 'time_by_schema.tex')


def algorithms_by_age_latex_table(dataframe, caption, label, tables_output_folder):
    """
    Generate a LaTeX table with the number of algorithms used by AGE_CATEGORY
    :param dataframe: Pandas DataFrame
    :param caption:   Caption of the table
    :param label:     Label of the table
    :return:          None
    """
    age_category_order = dataframe.groupby('AGE_GROUP')[
        'AGE_CATEGORY'].unique().explode().unique().tolist()  # Update with the desired order of AGE_GROUP

    # Convert SCHEMA_ID and AGE_GROUP to categorical data types with the specified order
    dataframe['AGE_CATEGORY'] = pd.Categorical(dataframe['AGE_CATEGORY'], categories=age_category_order, ordered=True)

    # Get the unique ALGORITHM_ID, ALGORITHM_DIMENSION, and AGE_CATEGORY
    algorithms_dataframe = dataframe[['ALGORITHM_ID', 'ALGORITHM_DIMENSION', 'AGE_CATEGORY']].drop_duplicates().dropna()

    # Calculate the total number of algorithms for each age category
    num_algorithms = algorithms_dataframe.groupby('AGE_CATEGORY', observed=False)['ALGORITHM_ID'].count().reset_index()
    num_algorithms.columns = ['AGE_CATEGORY', 'num_algorithms']

    # Calculate the counts of ALGORITHM_DIMENSION for each age category
    dimension_counts = algorithms_dataframe.groupby(['AGE_CATEGORY', 'ALGORITHM_DIMENSION'], observed=False).size().unstack(
        fill_value=0).reset_index()

    # Merge the dimension counts with the total number of algorithms
    full_df = num_algorithms.merge(dimension_counts, on='AGE_CATEGORY')

    final_df = full_df.copy()
    # Calculate the percentages for each ALGORITHM_DIMENSION
    for dim in [0, 1, 2]:
        final_df[dim] = final_df[dim].astype(str) + '/' + final_df['num_algorithms'].astype(str) + ' (' \
                        + (final_df[dim] / final_df['num_algorithms'] * 100).round().astype(int).astype(str) + '\%)'

    # Calculate the total counts and percentages for each ALGORITHM_DIMENSION across all age categories
    total_counts = full_df.drop(columns=['AGE_CATEGORY']).sum()
    total_percentages = (total_counts / total_counts['num_algorithms'] * 100).round()

    # Create the total row
    total_row = {'AGE_CATEGORY': r'\textbf{Total}', 'num_algorithms': total_counts['num_algorithms']}
    for dim in [0, 1, 2]:
        total_row[dim] = f"{int(total_counts[dim + 1])}/{int(total_counts['num_algorithms'])} ({int(total_percentages[dim + 1])}\%)"

    # Convert the total_row into a DataFrame and concatenate it with the original full_df
    total_df = pd.DataFrame(total_row, index=[0])
    final_df = pd.concat([final_df, total_df], ignore_index=True)

    # Specify the first row as the header with bold formatting
    header_row = [r'\textbf{Age category}', r'\textbf{Num. of unique algorithms}',
                  r'\multicolumn{1}{>{\centering\arraybackslash}m{2.2cm}}{\textbf{Num. of 0D algorithms}}',
                  r'\multicolumn{1}{>{\centering\arraybackslash}m{2.2cm}}{\textbf{Num. of 1D algorithms}}',
                  r'\multicolumn{1}{>{\centering\arraybackslash}m{2.2cm}}{\textbf{Num. of 2D algorithms}}']

    # Add the column widths to the LaTeX code
    ws = r'l>{\centering\arraybackslash}m{2.2cm}>{\raggedleft\arraybackslash}m{2.2cm}' \
         r'>{\raggedleft\arraybackslash}m{2.2cm}>{\raggedleft\arraybackslash}m{2.2cm}'

    # Generate LaTeX code with custom column widths
    latex_code = tabulate(final_df, headers=header_row, tablefmt='latex_raw', colalign=['c'] * len(header_row),
                          showindex=False)

    latex_code = latex_code.replace(r'{lllll}', r'{' + ''.join(ws) + '}')

    # Insert the \cmidrule command before the last row
    midrule = r'\cmidrule(lr){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}'
    latex_code = latex_code.replace(r'\textbf{Total}', midrule + r'\textbf{Total}')

    save_and_print_table(latex_code, caption, label, tables_output_folder, 'algorithms_by_age.tex')


def algorithms_by_schema_latex_table(dataframe, caption, label, tables_output_folder):
    """
    Generate a LaTeX table with the number of algorithms used by SCHEMA_ID
    :param dataframe: Pandas DataFrame
    :param caption:   Caption of the table
    :param label:     Label of the table
    :return:          None
    """
    # Retrieve unique values for SCHEMA_ID column
    unique_schema_ids = dataframe['SCHEMA_ID'].unique().tolist()

    # Initialise empty lists for counts
    unique_algorithm_counts = []
    zero_d_algorithm_counts = []
    one_d_algorithm_counts = []
    two_d_algorithm_counts = []

    # Iterate over unique SCHEMA_ID values
    for schema_id in unique_schema_ids:
        # Filter the DataFrame for the current SCHEMA_ID
        schema_data = dataframe[dataframe['SCHEMA_ID'] == schema_id]

        schema_data = schema_data[['ALGORITHM_ID', 'ALGORITHM_DIMENSION', 'SCHEMA_ID']].drop_duplicates().dropna()

        unique_algorithm_counts.append(schema_data['ALGORITHM_ID'].nunique())
        unique_algorithms = schema_data['ALGORITHM_ID'].nunique()

        # Count and percentage calculation for each algorithm dimension
        zero_d_count = len(schema_data[schema_data['ALGORITHM_DIMENSION'] == 0])
        zero_d_percentage = f"{zero_d_count}/{unique_algorithms} ({(zero_d_count / unique_algorithms * 100):.0f}\%)"
        zero_d_algorithm_counts.append(zero_d_percentage)

        one_d_count = len(schema_data[schema_data['ALGORITHM_DIMENSION'] == 1])
        one_d_percentage = f"{one_d_count}/{unique_algorithms} ({(one_d_count / unique_algorithms * 100):.0f}\%)"
        one_d_algorithm_counts.append(one_d_percentage)

        two_d_count = len(schema_data[schema_data['ALGORITHM_DIMENSION'] == 2])
        two_d_percentage = f"{two_d_count}/{unique_algorithms} ({(two_d_count / unique_algorithms * 100):.0f}\%)"
        two_d_algorithm_counts.append(two_d_percentage)

    # Create a DataFrame with the counts and percentages
    schema_counts_df = pd.DataFrame({
        'Schema': unique_schema_ids,
        'Num. of unique algorithms': unique_algorithm_counts,
        'Num. of 0D algorithms': zero_d_algorithm_counts,
        'Num. of 1D algorithms': one_d_algorithm_counts,
        'Num. of 2D algorithms': two_d_algorithm_counts
    })

    # Specify the first row as the header with bold formatting
    header_row = [r'\textbf{Schema}', r'\textbf{Num. of unique algorithms}',
                  r'\multicolumn{1}{>{\centering\arraybackslash}m{2.2cm}}{\textbf{Num. of 0D algorithms}}',
                  r'\multicolumn{1}{>{\centering\arraybackslash}m{2.2cm}}{\textbf{Num. of 1D algorithms}}',
                  r'\multicolumn{1}{>{\centering\arraybackslash}m{2.2cm}}{\textbf{Num. of 2D algorithms}}']

    # Add the column widths to the LaTeX code
    ws = r'c>{\centering\arraybackslash}m{2.2cm}>{\raggedleft\arraybackslash}m{2.2cm}' \
         r'>{\raggedleft\arraybackslash}m{2.2cm}>{\raggedleft\arraybackslash}m{2.2cm}'

    # Generate LaTeX code with custom column widths
    latex_code = tabulate(schema_counts_df, headers=header_row, tablefmt='latex_raw',
                          colalign=['c'] * len(header_row),
                          showindex=False)

    latex_code = latex_code.replace(r'{lllll}', r'{' + ''.join(ws) + '}')

    save_and_print_table(latex_code, caption, label, tables_output_folder, 'algorithms_by_schema.tex')


def time_by_category_latex_table(dataframe, category, caption, label, tables_output_folder, filename, std=False):
    """
    Generate a LaTeX table with time analysis data by a given category
    :param dataframe:            the pandas DataFrame
    :param category:             the category to group by
    :param caption:              the caption of the table
    :param label:                the label of the table
    :param tables_output_folder: the folder where to save the table
    :param filename:             the name of the file to save the table to
    :param std:                  Boolean indicating whether to include the standard deviation or not in the final table
    """
    if category == 'ARTEFACT':
        first_column_header = 'Interaction dimension'
        dimension_name = category + '_DIMENSION'
        category_name = category + '_TYPE'
    else:
        first_column_header = 'Age category'
        dimension_name = category + '_GROUP'
        category_name = category + '_CATEGORY'

    category_order = dataframe.groupby(dimension_name, observed=False)[category_name].unique().explode().unique().tolist()

    # Convert SCHEMA_ID and category to categorical data types with the specified order
    dataframe[category_name] = pd.Categorical(dataframe[category_name],
                                              categories=category_order,
                                              ordered=True)

    header_columns = [first_column_header, 'Mean', 'Min', 'Q1', 'Median', 'Q3', 'Max']
    table_orginal_format = r'{lllllll}'
    table_new_format = r'{lcccccc}'

    # Keep in the dataframe only the relevant columns
    # dataframe = dataframe[['STUDENT_ID', 'SCHEMA_ID', 'LOG_TIME', category_name]].copy()

    # # Group by category and SCHEMA_ID and calculate the mean time
    # schema_mean_times = dataframe.groupby(['SCHEMA_ID'], observed=False)['LOG_TIME'].mean().reset_index()
    #
    # # Merge the original DataFrame with the mean times DataFrame
    # grouped_times_total_filled = pd.merge(dataframe[['SCHEMA_ID', 'STUDENT_ID', 'LOG_TIME']],
    #                                       schema_mean_times, on=['SCHEMA_ID'],
    #                                       how='left')
    #
    # # Fill the missing values with the avg time
    # grouped_times_total_filled['LOG_TIME'] = grouped_times_total_filled['LOG_TIME_x'].fillna(
    #     grouped_times_total_filled['LOG_TIME_y'])
    #
    # grouped_times_total_filled.drop(columns=['LOG_TIME_x', 'LOG_TIME_y'], inplace=True)

    # Calculate the overall average, standard deviation, minimum, and maximum time to complete the test
    total_row = dataframe.groupby('STUDENT_ID', observed=False)['LOG_TIME'].sum().describe().to_frame()

    # remove the count and std columns
    total_row = total_row.drop('std')
    total_row = total_row.drop('count')

    total_row = total_row.T

    columns = ['Mean', 'Min', 'Q1', 'Median', 'Q3', 'Max']

    # Replace columns name in total_row with the list columns
    total_row.columns = columns

    # Apply to each column the format_time function
    total_row = total_row.applymap(format_time)

    total_row.reset_index(drop=True, inplace=True)

    # Add a new column at the beginning for the category with value r'\textbf{Total}'
    total_row.insert(0, category_name, r'\textbf{Total}')

    if category == 'AGE':
        grouped_times_total = dataframe.groupby(['STUDENT_ID'], observed=False)[
            'LOG_TIME'].sum().reset_index()
        grouped_times_total = grouped_times_total.merge(
            dataframe[['STUDENT_ID', category_name]],
            on='STUDENT_ID',
            how='left'
        )

        grouped_data = grouped_times_total.groupby(category_name, observed=False)['LOG_TIME']
        grouped_data_stats = grouped_data.describe()

    else:
        # grouped_times_total_filled = dataframe.merge(
        #     dataframe[['STUDENT_ID', 'SCHEMA_ID', category_name]],
        #     on=['STUDENT_ID', 'SCHEMA_ID'],
        #     how='left'
        # )

        grouped_times_total = dataframe.groupby(['STUDENT_ID', category_name], observed=False)[
            'LOG_TIME'].sum().reset_index()

        # Substitute the values 0 with NaN
        grouped_times_total['LOG_TIME'] = grouped_times_total['LOG_TIME'].replace(0, np.nan)

        grouped_data = grouped_times_total.groupby(category_name, observed=False)['LOG_TIME']
        grouped_data_stats = grouped_data.describe()

    # If the statistics contain a time equal to 0, make it equal to 1
    grouped_data_stats = grouped_data_stats.apply(lambda col: col.map(lambda x: 1 if x == 0 else x))

    # Remove the columns count and std
    grouped_data_stats = grouped_data_stats.drop(columns=['count', 'std'])

    # Rename using the elements in columns
    grouped_data_stats.columns = columns

    # Convert values from seconds to minutes and seconds and represent as strings
    grouped_data_stats = grouped_data_stats.applymap(format_time)

    # Transform the index into a column
    grouped_data_stats.reset_index(drop=False, inplace=True)

    # Convert grouped_data_stats to a list of DataFrames
    list_of_dfs = [grouped_data_stats, total_row]

    # Concatenate the list of DataFrames vertically
    merged_data = pd.concat(list_of_dfs, ignore_index=True)

    # RANDOM PLOT
    # statistics_DF = dataframe.groupby(category_name)['LOG_TIME'].agg(
    #     ['mean', 'min', 'max', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]).reset_index()
    # statistics_DF.columns = ['Category', 'Mean', 'Min', 'Max', 'Median', 'Q1', 'Q3']
    #
    # max_students = dataframe[category_name].value_counts().max()
    #
    # import matplotlib.pyplot as plt
    #
    # for category in statistics_DF['Category']:
    #     category_data = dataframe[dataframe[category_name] == category]
    #     category_data = category_data.sort_values(by='LOG_TIME').reset_index()
    #     # remove column called index
    #     category_data = category_data.drop(columns=['index'])
    #     # category_data = category_data.iloc[:1000]
    #     category_data = category_data.sample(max_students, replace=True)
    #
    #     plt.plot(category_data.index, category_data['LOG_TIME'], label=f'Category {category}')
    #
    # legend_text = []
    # for index, row in statistics_DF.iterrows():
    #     legend_text.append(
    #         f'Category {row["Category"]} - Mean: {format_time(row["Mean"])} Min: {format_time(row["Min"])} '
    #         f'Max: {format_time(row["Max"])} Median: {format_time(row["Median"])} '
    #         f'Q1: {format_time(row["Q1"])} Q3: {format_time(row["Q3"])}')
    #
    # plt.legend(legend_text, loc='upper left')
    #
    # # Customize labels and title
    # plt.xlabel('STUDENT_ID')
    # plt.ylabel('LOG_TIME')
    # plt.title('Artefact Category vs. LOG_TIME')
    #
    # plt.show()

    # Create the overall table headers
    header_row = []
    for cell in header_columns:
        header_row.append(r'\textbf{{{}}}'.format(cell))

    # Generate the LaTeX table for the overall data
    latex_code = tabulate(merged_data, headers=header_row, tablefmt='latex_raw', showindex=False)
    latex_code = latex_code.replace(table_orginal_format, table_new_format)

    # Insert the \cmidrule command before the last row
    midrule = r'\cmidrule(lr){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}'
    latex_code = latex_code.replace(r'\textbf{Total}', midrule + r'\textbf{Total}')

    save_and_print_table(latex_code, caption, label, tables_output_folder, filename)
