import pandas as pd
import os
from .connector import PostgresConnector,SQLiteConnector
from dotenv import load_dotenv
load_dotenv()
import psycopg2
import json
import logging

logging.basicConfig(level=logging.INFO)


class DataReader:
    def __init__(self):
        self.data = None
        self.choice= None
        self.host=None
        self.port=None
        self.dbname=None
        self.table_name=None
        self.password=None
        self.sqlite_file_path=None
        self.csv_file_path=None
        self.xlsx_file_path=None
    def connecttosqlite(self):
        sqliteconnector=SQLiteConnector(db_folder=self.sqlite_file_path)
        sqliteconnector.connect()

        df=sqliteconnector.sqlite_to_dataframe(table_name=self.table_name)
        return df
    def read_data(self,**kwargs):
        '''file_path : path for csv or xlsx files.
           db_path : path for database.
           table_name : name of the table in question for the current database.
           host : host name for postgres
           dbname : database name for postgres
           user : user name in postgres
           password : password for postgres
        '''
        source_type=self.choice
        if source_type == 'csv':
            self.data = pd.read_csv(os.path.join(self.csv_file_path,f'{self.table_name}.csv'), encoding='utf-8')
        elif source_type == 'xlsx':
            self.data = pd.read_excel(os.path.join(self.xlsx_file_path,f'{self.table_name}.xlsx'),sheet_name=kwargs['selected_sheet'])
        elif source_type == 'sqlite':
            
            self.data=self.connecttosqlite()
        elif source_type == 'postgres':
            self.data = self.read_from_postgres()
        else:
            raise ValueError("Invalid file type. Please choose from 'csv', 'excel', 'sqlite', or 'postgres'.")        
        return self.data

    def read_from_postgres(self):
        postgresconnector=PostgresConnector(host=self.host,port=self.port,dbname=self.dbname,user=self.user,password=self.password)
        postgresconnector.connect()

        try:
            df=postgresconnector.extract_table_to_dataframe(table_name=self.table_name,batch_size=10000)
            return df
        except psycopg2.Error as e:
            logging.error(f"An error occurred: {e}")
            return None
    
    def getuserchoice(self,choice):
        if choice=='csv' or choice=='xlsx' or choice=='postgres' or choice=='sqlite':
            self.choice=choice
        else:
            raise ValueError('Invalid Choice')
    
    def getcredentials(self):
        config_file=os.path.join('pages','src','Database','config.json')
        with open(config_file,'r') as file:
            config_data=json.load(file)

        self.sqlite_file_path = config_data.get("sqlite_file_path")
        self.host = config_data.get("postgres_host")
        self.port = config_data.get("postgres_port")
        self.dbname = config_data.get("postgres_dbname")
        self.user = config_data.get("postgres_user")
        self.password = config_data.get("postgres_password")
        self.table_name = config_data.get("current_table_name")
        self.csv_file_path = config_data.get("csv_file_path")
        self.xlsx_file_path = config_data.get("xlsx_file_path")
