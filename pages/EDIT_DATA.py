import streamlit as st
import pandas as pd
from datetime import datetime
from AutoClean.autoclean import AutoClean
from streamlit_extras import colored_header
import numpy as np


class DataFrameModifier:
    def __init__(self):
        self.data = None
        if 'modified_data' not in st.session_state:
            st.session_state['modified_data'] = None
        self.modified_data=None
        self.restore_point_data=None
        st.session_state.inplace=None
    

    @st.experimental_fragment
    def display(self):
        if self.data is not None:
            if st.session_state.modified_data is not None:
                    self.data=st.session_state.modified_data
            st.data_editor(self.data, use_container_width=True, key=f"{self.data}")
            
        else:
            st.write("No data to display.")







    def add_row(self):
        if self.data is not None:
            row = {}
            columns = self.data.columns

            

            with st.sidebar.form(key=f"{self.data}" , border=True):
                with st.expander('Add row'):
                    for column in columns:
                        if self.data[column].dtype == 'object':
                            row[column] = st.text_input(f"Enter {column} value", key=column)
                        else:
                            row[column] = st.number_input(f"Enter {column} value", key=column, value=1)
                    
                    submit=st.form_submit_button("submit")
            if submit:
                if st.session_state.modified_data is not None:
                    self.data=st.session_state.modified_data
                new_row = pd.DataFrame(row, index=[len(self.data)])
                self.modified_data = pd.concat([self.data, new_row])
                st.session_state.modified_data=self.modified_data
                self.data=st.session_state.modified_data
                print(st.session_state.modified_data)
                st.success("Row Added successfully.")
                # self.display()
                st.rerun()

    def remove_row(self):
        if self.data is not None:

           

                with st.sidebar.form(key=f"{self.data} + 2" , border=True):

                    with st.expander('Remove Row'):
                        st.write('delete by index')
                        indices=st.multiselect('index',options=np.arange(self.data.shape[0]))
                    
                        submit=st.form_submit_button("submit")
                if submit:
                    if st.session_state.modified_data is not None:
                        self.data=st.session_state.modified_data
                    self.modified_data = self.data.drop(index=indices,axis=0)
                    st.session_state.modified_data=self.modified_data
                    self.data=st.session_state.modified_data
                    print(st.session_state.modified_data)
                    st.success("Row Deleted successfully.")
                    st.rerun()
                    # self.display()


    def cleanData(self, mode='auto', duplicates='auto', missing_num='auto', missing_categ='auto', encode_categ=False, extract_datetime=False, outliers=False,logfile=True, verbose=False, **kwargs):
            if st.session_state.modified_data is not None:
                self.data = st.session_state.modified_data

            print(mode,missing_categ,extract_datetime,encode_categ,missing_num)
            if 'outlier_param' in kwargs and  kwargs['outlier_param']==1:
                outlier_param=kwargs['outlier_param']
            else:
                outlier_param=1.5

            
            with st.sidebar.expander('Clean Data'):
                if st.button('Clean (inplace=True)'):
                    pipeline = AutoClean(self.data, mode=mode, duplicates=duplicates, missing_num=missing_num, missing_categ=missing_categ, encode_categ=encode_categ, extract_datetime=extract_datetime, outliers=outliers, outlier_param= outlier_param, logfile=logfile, verbose=verbose)
                    st.session_state.modified_data = pipeline.output
                    self.data = st.session_state.modified_data
                    st.session_state.inplace = True
                elif st.button('Clean (inplace=False)'):
                    pipeline = AutoClean(self.data, mode=mode, duplicates=duplicates, missing_num=missing_num, missing_categ=missing_categ, encode_categ=encode_categ, extract_datetime=extract_datetime, outliers=outliers, outlier_param= outlier_param, logfile=logfile, verbose=verbose)
                    st.session_state.inplace = False

            if st.session_state.inplace == False:
                st.data_editor(pipeline.output, use_container_width=True)
            # elif st.session_state.inplace == True:
            #     st.data_editor(st.session_state.modified_data, width=0)
        
    def drop_column(self):
        if st.session_state.modified_data is not None:
            self.data=st.session_state.modified_data
        
        with st.sidebar.expander('Drop Columns'):
            columns=self.data.columns
            selected_columns=st.multiselect('choose columns', columns)

            if st.button('Drop Columns'):
                for columns in selected_columns:
                    st.session_state.modified_data=self.data.drop(selected_columns,axis=1)
                    self.data=st.session_state.modified_data
                    
    def restore(self):

        if st.button('Restore to initial state'):
            self.data=self.restore_point_data
            st.session_state.modified_data=self.restore_point_data
            st.rerun()


    def download_csv(self):
        if self.data is not None:
            csv_data = self.data.to_csv(index=False)
            st.download_button(label="Download CSV", data=csv_data, file_name="modified_data.csv", mime="text/csv")
        else:
            st.write("No data available to download.")

    def upload_csv(self, uploaded_file):
        if uploaded_file is not None:
            self.data = pd.read_csv(uploaded_file, encoding='iso-8859-1')
            self.restore_point_data=self.data.copy()
            st.success("File uploaded successfully.")
        else:
            st.error("Please upload a CSV file.")
        
        return self.data

data_modifier = DataFrameModifier()
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    data_modifier.upload_csv(uploaded_file)


def run():

    if data_modifier.data is not None:
        
        mode = st.sidebar.selectbox("Mode", ['auto', 'manual'])
        if mode=='manual':
            duplicates = st.sidebar.selectbox("Handle Duplicates", ['auto', False])
            missing_num = st.sidebar.selectbox("Handle Numerical Missing Values", ['auto', 'linreg', 'knn', 'mean', 'median', 'most_frequent', 'delete', False])
            missing_categ = st.sidebar.selectbox("Handle Categorical Missing Values", ['auto', 'logreg', 'knn', 'most_frequent', 'delete', False])
            encode_categ = st.sidebar.selectbox("Encode Categorical Features",[['auto'],['onehot'],['label'],False])
            extract_datetime = st.sidebar.selectbox("Extract Datetime Features", [False, 'D', 'M', 'Y', 'h', 'm', 's'])
            outliers = st.sidebar.selectbox("Handle Outliers", [False, 'winz', 'delete'])
            outlier_param = None
            if outliers and outliers != False:
                outlier_param = st.sidebar.number_input("Outlier Parameter", value=1.5)
            verbose = st.sidebar.checkbox("Verbose")
        elif mode=='auto':
            duplicates=False
            missing_num='auto'
            missing_categ='auto'
            encode_categ=False
            outliers=False
            outlier_param=1
            verbose=False
            extract_datetime=False
   
        print(outlier_param)
        data_modifier.cleanData(mode, duplicates, missing_num, missing_categ, encode_categ, extract_datetime, outliers,verbose, outlier_param=outlier_param)

        if st.checkbox("Edit Data"):
            col1,col2,col3=st.columns(3)
            with col1:
                data_modifier.add_row()
                if st.session_state.modified_data is not None:
                    data_modifier.restore()
            with col2:
                data_modifier.remove_row()
                data_modifier.drop_column()
        
                
        

            

    data_modifier.download_csv()

if __name__=='__main__':
    colored_header.colored_header("Edit Data Before EDA","You can manipulate data here if you ever need to and then perform eda on the same data.",color_name='blue-green-90')
    data_modifier.display()
    run()


