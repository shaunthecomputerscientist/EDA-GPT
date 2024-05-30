from textwrap import dedent
import sys
from pages.src.Tools.tools import Calculator
# sys.path.insert(0,'../../../vendor/dependencies/crewAI')
# from vendor.dependencies.crewAI.crewai import Agent,Crew,Task
# sys.path.pop(0)
from crewai import Agent,Task,Crew


def structured_tasks(agent, table_name, formatted_data, questions):
    data_summary_planner = """
    PROVIDE A COMPREHENSIVE OVERVIEW OF THE DATA AFTER CAREFULLY ANALYZING IT.
    The Table Name: {table_name}
    DATA OVERVIEW:
    - Number of rows:
    - Number of columns:
    - Column names and their data types:
    - Provide a preview of the first few rows of the dataset.
    SUMMARY:
    - Briefly describe the purpose and contents of the dataset, including its source and any relevant background information.
    - Highlight any notable features, patterns, or relationships observed in the data.
    - Mention any potential issues, anomalies, or limitations with the data (e.g., missing values, outliers).
    - Discuss how this data could be used for further analysis or decision-making.

    DATA: {formatted_data}
    """

    statistics_summary_planner = """
    PROVIDE SUMMARY STATISTICS ABOUT THE DATA AND DERIVE KEY INSIGHTS FROM THEM.
    The Table Name: {table_name}
    SUMMARY STATISTICS:
    - For each numeric column, provide:
    - Mean
    - Median
    - Standard deviation
    - Minimum value
    - Maximum value
    - Percentiles (e.g., 25th, 50th, 75th)
    - For each categorical column, provide:
    - Unique values
    - Frequencies of each unique value
    - Mode and its frequency
    - Identify any outliers or extreme values in the data, using appropriate statistical methods (e.g., IQR, z-scores).

    KEY INSIGHTS:
    - Discuss any interesting patterns or relationships revealed by the statistics.
    - Highlight any surprising or unexpected findings.
    - Mention any limitations or caveats of the statistics (e.g., skewness, kurtosis).
    - Suggest how these insights could inform further analysis or decision-making.
    DATA: {formatted_data}
    """

    null_missing_unique_values = """
    PROVIDE DETAILED INSIGHTS INTO NULL, MISSING, AND UNIQUE VALUES IN THE DATA.

    The Table Name: {table_name}

    NULL AND MISSING VALUES:
    - For each column, list the number and percentage of null/missing values.
    - Visualize the distribution of missing values across the dataset.
    - Identify any patterns or relationships with null values (e.g., certain columns or rows have more nulls).
    - Discuss potential reasons for missing data and how it may impact analysis.

    UNIQUE VALUES / VALUE COUNTS:
    - For each column, list the number of unique values and their frequencies.
    - Identify any columns with a small number of unique values (e.g., binary, categorical).
    - Visualize the distribution of unique values for key columns.

    INSIGHTS:
    - Discuss how null/missing values and unique values could affect data quality and analysis.
    - Suggest ways to handle null values (e.g., imputation, dropping rows/columns).
    - Mention any interesting findings related to unique values.
    - Recommend further investigation or data cleaning based on these insights.

    DATA: {formatted_data}
    """

    column_analysis_provider = """
    ANALYZE IMPORTANT COLUMNS IN THE DATA AND DERIVE INSIGHTS FROM THEM.
    COLUMN ANALYSIS:
    - For each important column, provide a brief description of its contents and purpose
    - Analyze the distribution of values in the column (e.g. mean, median, unique values, outliers)
    - Identify any interesting patterns, relationships, or insights revealed by the column
    - Discuss how the column could be used in analysis or modeling

    INSIGHTS:
    - Summarize the key takeaways from analyzing the important columns
    - Discuss how the columns relate to each other and the overall dataset
    - Suggest additional columns or data that could provide further insights
    - Recommend how this column-level analysis could inform further investigation or modeling

    DATA: {formatted_data}
    """

    overall_understanding = """
    PROVIDE AN OVERALL UNDERSTANDING OF THE DATA AS A DATA ANALYST IN APPROXIMATELY 2000 WORDS.

    Title : Overall analysis

    INTRODUCTION:
    - Briefly introduce the dataset and its purpose
    - Mention the source and any relevant background information

    DATA QUALITY AND CLEANING:
    - Discuss the overall quality of the data (e.g. missing values, outliers, errors)
    - Describe any data cleaning or preprocessing steps taken
    - Mention any limitations or caveats of the data

    EXPLORATORY DATA ANALYSIS:
    - Provide summary statistics and visualizations for key variables
    - Identify any interesting patterns, trends, or relationships in the data
    - Discuss how the data could be used to answer specific questions or test hypotheses

    INSIGHTS AND RECOMMENDATIONS:
    - Summarize the key insights gained from analyzing the data
    - Discuss how these insights could inform decision-making or further analysis
    - Recommend additional data, analysis, or modeling that could provide further insights
    - Mention any limitations or assumptions of the analysis

    CONCLUSION:
    - Restate the main takeaways and their significance
    - Discuss potential applications or implications of the analysis
    - Suggest future directions for research or investigation

    DATA: {formatted_data}
    """

    tasks=[]
    data_summary_task=Task(
                description=dedent(data_summary_planner).format(table_name=table_name, formatted_data=formatted_data),
                agent=agent,
                async_execution=False,
                expected_output=dedent(f"""The output must contain summary and understanding of the data.\n"""),
            )
    
    data_statistics_analyzer=Task(
                description=dedent(statistics_summary_planner).format(table_name=table_name,formatted_data=formatted_data),
                agent=agent,
                async_execution=False,
                expected_output=dedent(f"""provide summary statistics tables and insights on statistics.\n"""),
                )
    
        
    important_values_analyzer=Task(
                description=dedent(null_missing_unique_values).format(table_name=table_name,formatted_data=formatted_data),
                agent=agent,
                async_execution=False,
                expected_output=dedent(f"""Provide Insights On All these Important values..."""),
                )
    

    columns_analyzer=Task(
                description=dedent(column_analysis_provider).format(formatted_data=formatted_data),
                agent=agent,
                async_execution=False,
                expected_output=dedent(f"""Extract important columns and analyze their dtypes and what they signify providing comprehensive understanding of the columns.\n"""),
                )
    
    data_analyzer=Task(
        description=dedent(overall_understanding).format(formatted_data=formatted_data),
        agent=agent,
        async_execution=False,
        expected_output=dedent(f"""Extract important columns and analyze their dtypes and what they signify providing comprehensive understanding of the columns.\n"""),
    )

    subtasks=[data_summary_task,data_statistics_analyzer,important_values_analyzer, columns_analyzer, data_analyzer]

    

    if questions!="":
        user_query_answer=Task(
        description=dedent("Answer following user questions from the data given.\n Data : {formatted_data} \n\n {questions}").format(formatted_data=formatted_data,questions=questions),
        agent=agent,    
        async_execution=False,
        expected_output=dedent('Answer User Questions In Sequence. Title: USER QUESTIONS ANSWERED\n\n')
    )
        tasks=subtasks+[user_query_answer]

    

    else:

        tasks=subtasks

    
    return tasks
    
    
    
