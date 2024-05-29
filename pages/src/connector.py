import psycopg2
import pandas as pd
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO)


class PostgresConnector:
    def __init__(self, host, port, dbname, user, password):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.conn = None
    
    def connect(self):
        # Construct the connection string
        conn_string = f"host='{self.host}' port='{self.port}' dbname='{self.dbname}' user='{self.user}' password='{self.password}'"
        
        try:
            # Connect to the PostgreSQL database
            self.conn = psycopg2.connect(conn_string)
            logging.info("Connected to PostgreSQL successfully!")
            return True
        except psycopg2.Error as e:
            logging.info(f"Unable to connect to the database: {e}")
            return False
    
    def extract_table_to_dataframe(self, table_name, batch_size=10000):
        if not self.conn:
            logging.info("Connection not established.")
            return None
        
        try:
            # Create a cursor for streaming data from PostgreSQL
            cursor = self.conn.cursor(name='streaming_cursor')
            cursor2 = self.conn.cursor()
            
            # Query to select all data from the specified table
            query = f"SELECT * FROM {table_name};"
            columns_query = f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name}' ORDER BY ordinal_position"
            
            # Execute the query
            cursor.execute(query)
            cursor2.execute(columns_query)

            # Fetch data in batches
            rows = cursor.fetchmany(batch_size)
            columns = cursor2.fetchall()
            data = []
            while rows:
                data.extend(rows)
                rows = cursor.fetchmany(batch_size)
            
            # Close the cursor
            cursor.close()
            
            # Convert data to pandas DataFrame
            df = pd.DataFrame(data, columns=[row[0] for row in columns])
            return df
        except psycopg2.Error as e:
            logging.info(f"Error extracting data from the table: {e}")
            
            raise ValueError(f"Could not find table {table_name}")
        finally:
            # Close the cursor and the connection to the database
            cursor2.close()
            self.conn.close()

class SQLiteConnector:
    def __init__(self, db_folder):
        self.db_folder = db_folder
        self.db_file = None
        self.conn = None
    
    def connect(self):
        '''Connect to the SQLite database'''
        try:
            # Get the path to the SQLite database file
            db_files = os.listdir(self.db_folder)
            if db_files:
                self.db_file = os.path.join(self.db_folder, db_files[0])
            else:
                logging.warning("No SQLite database file found.")
                return False
            
            # Establish connection
            self.conn = sqlite3.connect(self.db_file)
            logging.info("Connected to SQLite database successfully!")
            return True
        except sqlite3.Error as e:
            logging.error(f"Unable to connect to the SQLite database, check your credentials: {e}")
            return False
    
    def sqlite_to_dataframe(self, table_name):
        '''Convert SQLite table to pandas DataFrame'''
        if not self.conn:
            logging.warning("Connection not established.")
            return None
        
        try:
            # Query to select all data from the specified table
            query = f"SELECT * FROM {table_name};"
            
            # Read data from SQLite into a pandas DataFrame
            df = pd.read_sql_query(query, self.conn)
            
            return df
        except sqlite3.Error as e:
            logging.error(f"An error occurred while querying the table: {e}")
            return None
        finally:
            # Close the connection to the database
            if self.conn:
                self.conn.close()