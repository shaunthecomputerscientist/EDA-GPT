import streamlit as st
from .data import DataReader
import pandas as pd
import os
from streamlit_extras.dataframe_explorer import dataframe_explorer
import json
import streamlit as st
import logging
logging.basicConfig(level=logging.INFO)
# from streamlit_server_state import server_state


class DataFrameEnvironment:
    def __init__(self):
        self.data = None
        self.dataframe_placeholder=st.empty

    def load_data(self, dataframe):
        self.data = dataframe
    @st.fragment
    def display(_self, data):
        if _self.data is not None:
            filtered_df=dataframe_explorer(_self.data)
            _self.dataframe_placeholder=st.dataframe(filtered_df , use_container_width=True)
        else:
            _self.dataframe_placeholder=st.empty


    def add_row(self):
        if self.data is not None:
            row={}
            columns=self.data.columns
            for column in columns:
                if self.data[column].dtype=='object':
                    row[column]=st.text_input(f"Enter {column} value" , key=column)
                else:
                    row[column]=st.number_input(f"Enter {column} value" , key=column)

            if st.button("apply changes"):
                new_row=pd.DataFrame(row, index=[len(self.data)])
                self.data=pd.concat([self.data,new_row])
                self.dataframe_placeholder.dataframe(self.data, use_container_width=True)

        return self.data



class DataManager:
    def __init__(self, config_data, config_file_path):
        self.table_name=None
        self.reader=DataReader()
        self.config_data=config_data
        self.config_file_path=config_file_path
    def postgres_data(self):
        st.write("Please enter PostgreSQL credentials:")
        host = st.text_input("Host" , value='127.0.0.1')
        port = st.number_input("Port", value=5432)
        dbname = st.text_input("Database Name", value='DATABASE NAME')
        user = st.text_input("Username" , value='PGADMIN USERNAME')
        password = st.text_input("Password", type="password" , value='PGADMIN PASSWORD')
        table_name = st.text_input("Table Name")

        if host and port and dbname and user and password and table_name:
            config_file_path = os.path.join('pages','src', 'Database', 'config.json')
            with open(config_file_path, 'r') as file:
                config_data = json.load(file)

            config_data["postgres_host"] = host
            config_data["postgres_port"] = port
            config_data["postgres_dbname"] = dbname
            config_data["postgres_user"] = user
            config_data["postgres_password"] = password
            config_data["current_table_name"] = table_name

            with open(config_file_path, 'w') as file:
                json.dump(config_data, file, indent=4)

            self.reader.getcredentials()
            self.reader.getuserchoice('postgres')
            return self.reader.read_data()
        else:
            return None

    def sqlite_data(self):

        self.reader.getcredentials()
        self.reader.getuserchoice('sqlite')
        return self.reader.read_data()

    def csv_data(self):
        self.reader.getcredentials()
        self.reader.getuserchoice('csv')
        return self.reader.read_data()

    def xlsx_data(self):
        self.reader.getcredentials()
        self.reader.getuserchoice('xlsx')
        sheet_names=(pd.ExcelFile(os.path.join(self.config_data['xlsx_file_path'],f'{self.table_name}.xlsx'))).sheet_names
        selected_sheet=st.selectbox("Select Sheet",sheet_names)
        if selected_sheet:
            return self.reader.read_data(selected_sheet=selected_sheet)


    def get_data(self):
        user_choice = st.selectbox("Choose data source:", ( 'CSV', 'SQLITE','POSTGRES','XLSX'))

        if user_choice.lower() == 'postgres':
            return self.postgres_data()
        elif user_choice.lower() in ['sqlite', 'csv', 'xlsx']:
            type={'sqlite':['sqlite','db'],'xlsx':['xlsx'],'csv':['csv']}
            uploaded_file = st.file_uploader(f"Upload {user_choice.upper()} File", accept_multiple_files=False , type=type[user_choice.lower()])

            
            return self.process_uploaded_file(uploaded_file, user_choice.lower())


    def process_uploaded_file(self, uploaded_file, file_type):
        
        # self._reset_data()
        
        if uploaded_file is not None:
            self.table_name = self.get_table_name(file_type, uploaded_file)
            self.save_uploaded_file(uploaded_file, file_type, self.table_name)
            if self.table_name is not None:
                return getattr(self, f"{file_type}_data")()
        return None

    def get_table_name(self, file_type, uploaded_file):

        if file_type == 'sqlite':
            self.table_name = st.text_input("Enter Table Name:")
            if self.table_name:
                return self.table_name

        elif file_type in ['csv', 'xlsx']:
            table_name = os.path.splitext(uploaded_file.name)[0]
            self.table_name=table_name
            return self.table_name

    def save_uploaded_file(self, uploaded_file, file_type, table_name):
        

        directory = self.config_data[f"{file_type}_file_path"]
        self.remove_existing_files(directory)

        # file_path = f"{directory}//{uploaded_file.name}"
        file_path = os.path.join(directory,uploaded_file.name)
        # logging.info("file path",file_path)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        if table_name:
            self.config_data["current_table_name"] = table_name
            with open(self.config_file_path, 'w') as file:
                json.dump(self.config_data, file, indent=4)

    def remove_existing_files(self, directory):
        file_list = os.listdir(directory)
        for file_name in file_list:
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    def _reset_data(self):
        self.table_name=None
        self.reader=DataReader()






