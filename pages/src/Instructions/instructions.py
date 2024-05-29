import streamlit as st
import streamlit as st
import plotly.express as px
import pandas as pd

def instructions():
    st.title("Welcome to EDA GPT: Your Data Analysis Companion")
    

    st.write("## Introduction")
    st.write("Welcome to EDA GPT, your one-stop solution for all your data analysis needs.")
    st.divider()

    st.write("### How to Use the App")
    st.write("1. **Structured Data Analysis**: ")
    st.write("   - Utilize EDA GPT to analyze structured data in CSV, XLSX, and SQLite formats.")
    st.write("   - Connect your PostgreSQL database to extract and analyze data seamlessly.")
    st.write("   - Provide additional context about your data for better analysis through the description input field.")
    st.write("   - Elaborate on the desired outcome to enhance analysis accuracy.")
    st.write("   - When interacting with EDA GPT, include metadata about the data such as data source, collection methods, and any preprocessing steps applied. This metadata helps EDA GPT make better decisions during analysis.")

    st.write("2. **Graph Generation**: ")
    st.write("   - Generate graphs effortlessly by asking EDA GPT about specific graphs and data.")
    st.write("   - Access the generated code for each graph to verify and fine-tune accordingly.")
    st.write("   - Benefits include saving time on manual graph code writing and focusing on higher-level analysis tasks.")
    st.write("   - When requesting graphs, provide clear instructions and context to EDA GPT. This could include specifying the type of graph desired (e.g., bar chart, scatter plot) and the variables to be plotted.")

    st.write("3. **Analysis Questions**: ")
    st.write("   - Post initial EDA, ask analysis questions atop the EDA report.")
    st.write("   - Gain insights through Plotly graphs on numerical and categorical columns, alongside an EDA visualization report for further analysis.")
    st.write("   - While formulating analysis questions, consider the specific aspects of your data you want to explore. This could involve investigating correlations between variables, identifying outliers, or uncovering patterns in the data.")
    st.write('')
    st.write("- **Surprisingly, EDA GPT performs well compared to pandas ai on the following benchmarks. The scores are normalized and scaled to be out of 100**")




    # Assuming you have performance data for EDA GPT and PandasAI
    # Example data
    performance_data = {
        'Tool': ['EDA GPT', 'PandasAI'],
        'Accuracy': [90, 85],  # Example accuracy scores (out of 100)
        'Speed': [92, 90],     # Example speed scores (out of 100)
        'Complex Queries': [90, 70]  # Example scores for handling complex queries (out of 100)
    }

    # Create DataFrame from performance data
    df = pd.DataFrame(performance_data)

    # Melt DataFrame for easier plotting
    df_melted = pd.melt(df, id_vars='Tool', var_name='Benchmark', value_name='Score')

    # Create bar plot
    fig = px.bar(df_melted, x='Tool', y='Score', color='Benchmark',
                barmode='group', title='📊Comparison of EDA GPT and PandasAI Performance')

    # Show plot
    st.plotly_chart(fig)

    st.write("4. **LLMs (Large Language Models)**: ")
    st.write("   - Choose from a variety of LLMs based on your dataset characteristics.")
    st.write("   - Opt for LLMs with large context windows like GPT-4 and CLAUDE-3 for extensive datasets (premium feature).")
    st.write("   - When selecting an LLM, consider factors such as the size of your dataset, the complexity of the analysis, and the level of detail required in the responses.")

    st.write("5. **Unstructured Data Analysis**: ")
    st.write("   - Analyze unstructured PDF data efficiently.")
    st.write("   - Support for other formats like XML and JSON is underway.")
    st.write("   - Provide detailed descriptions to enhance LLM decision-making.")
    st.write("   - Analyze images with integrated vision models.")
    st.write("   - When analyzing unstructured data, provide as much context as possible to assist the LLM in understanding the content. This could involve describing the layout of the document, specifying any key terms or topics, and highlighting any specific sections of interest.")

    st.write("6. **Multimodal Search**: ")
    st.write("   - EDA GPT searches answers from diverse sources including Wikipedia, Arxiv, DuckDuckGo, and web scrapers.")
    st.write("   - Access Internet-based information and analyze images using vision models.")
    st.write("   - Specify search space limitations if required.")
    st.write("   - When conducting a multimodal search, specify any constraints or preferences to narrow down the search results. This could include specifying the types of sources to include or exclude, setting a time frame for the search, or indicating the level of relevance or credibility required.")

    st.write("7. **Structured Data Analysis**:")
    st.write("Use EDA GPT to extract and analyze tables from PDFs and images, enabling further analysis within the structured data section.")
    st.write("   - When analyzing structured data, provide metadata such as data source, collection methods, preprocessing steps, and any relevant domain knowledge. This metadata assists EDA GPT in making better decisions during analysis.")


    st.write("8. **Data Cleaning and Editing**: ")
    st.write("   - Use EDA GPT to clean and edit your data using various methods such as imputation (e.g., logistic regression, k-nearest neighbors, most frequent value).")
    st.write("   - Explore different data cleaning techniques and select the most suitable ones based on your data characteristics and analysis goals.")
    st.write("   - Benefit from automated data cleaning processes provided by EDA GPT, saving time and effort in preparing your data for analysis.")





