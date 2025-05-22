import requests
import json
import pandas as pd
import csv
import psycopg2

#Extraction layer 

url = "https://api.rentcast.io/v1/properties/random?limit=100000"

headers = {
    "accept": "application/json",
    "X-Api-Key": "3c2248e54d1949dc9e344f1382415088"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    
    # Save the data to a file
    filename = "propertyrecords.json"
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data successfully saved to {filename}")
else:
    print(f"Request failed with status code: {response.status_code}")


#Read into a dat frame 
propertyrecords_df = pd.read_json("propertyrecords.json")


#Transformation layer
#convert dictionary colum to string
propertyrecords_df['features'] = propertyrecords_df['features'].apply(json.dumps)
propertyrecords_df['taxAssessments'] = propertyrecords_df['taxAssessments'].apply(json.dumps)
propertyrecords_df['propertyTaxes'] = propertyrecords_df['propertyTaxes'].apply(json.dumps) 
propertyrecords_df['history'] = propertyrecords_df['history'].apply(json.dumps)
propertyrecords_df['owner'] = propertyrecords_df['owner'].apply(json.dumps)
propertyrecords_df['hoa'] = propertyrecords_df['hoa'].apply(json.dumps)

# Check that all transformed columns are now strings
transformed_columns = ['features', 'taxAssessments', 'propertyTaxes', 'history', 'owner','hoa']

for col in transformed_columns:
    unique_types = propertyrecords_df[col].apply(type).unique()
    print(f"{col} types: {unique_types}")



# Check that all transformed columns are now strings
transformed_columns = ['features', 'taxAssessments', 'propertyTaxes', 'history', 'owner','hoa']

for col in transformed_columns:
    unique_types = propertyrecords_df[col].apply(type).unique()
    print(f"{col} types: {unique_types}")


# Only convert dictionaries to JSON strings if not already strings
import json

def safe_serialize(val):
    return json.dumps(val) if isinstance(val, dict) else val

propertyrecords_df['features'] = propertyrecords_df['features'].apply(safe_serialize)


propertyrecords_df.info()



#transformation layer 
#2nd step replace Nan values with appriate defaults or remove row/colums as ncessary.
propertyrecords_df.fillna({
    'addressLine2': 'unknown',
    'county': 'unknown',
    'propertyType': 'unknown',
    'bedrooms': 0.0,
    'bathrooms': 0.0,
    'squareFootage': 0.0,
    'lotSize': 0.0,
    'yearBuilt': 0.0,
    'assessorID': 'unknown',
    'legalDescription': 'unknown',
    'subdivision': 'unknown',
    'zoning': 'unknown',
    'lastSaleDate': 'unknown',
    'lastSalePrice': 0.0,
    'ownerOccupied': 0.0
}, inplace=True)


# To list the columns of the DataFrame
propertyrecords_df.columns


# Define the columns for dim_location
dim_location_columns = propertyrecords_df[['city', 'state', 'zipCode', 'county']].drop_duplicates().reset_index(drop=True)

# Assign unique 'location_id' to each unique location record
dim_location_columns['location_id'] = dim_location_columns.index + 1
dim_location_columns = dim_location_columns.set_index('location_id')

# View the dim_location table
print(dim_location_columns.head())




# Reset the index to make 'location_id' a column
dim_location_columns = dim_location_columns.reset_index()

# Check the updated columns of dim_location_columns
print(dim_location_columns.head())



# Define the columns for dim_sale
dim_sale_columns = propertyrecords_df[['lastSaleDate', 'lastSalePrice']].drop_duplicates().reset_index(drop=True)

# Assign unique 'sales_id' to each unique sale record
dim_sale_columns['sales_id'] = dim_sale_columns.index + 1
dim_sale_columns = dim_sale_columns.set_index('sales_id')

# View the dim_sale table
print(dim_sale_columns.head())


# Reset the index to make 'sales_id' a column
dim_sale_columns = dim_sale_columns.reset_index()
# Check the updated columns of dim_location_columns
print(dim_sale_columns.head())







# Only convert dictionaries to JSON strings if not already strings
import json

def safe_serialize(val):
    return json.dumps(val) if isinstance(val, dict) else val

propertyrecords_df['features'] = propertyrecords_df['features'].apply(safe_serialize)






import json

def clean_features(val):
    try:
        # Unpack multiple layers of string-encoded JSON
        while isinstance(val, str):
            val = json.loads(val)
        return json.dumps(val)  # Properly serialized once
    except:
        return json.dumps({})  # fallback for invalid entries

# Apply to DataFrame
propertyrecords_df['features'] = propertyrecords_df['features'].apply(clean_features)

# Now generate dim_features again
dim_features = (
    propertyrecords_df[['features']]
    .drop_duplicates()
    .reset_index(drop=True)
)

dim_features['features_id'] = dim_features.index + 1
dim_features = dim_features.set_index('features_id')

# Merge back into main DataFrame
propertyrecords_df = propertyrecords_df.merge(
    dim_features.reset_index(), on='features', how='left'
)

# Preview cleaned dim_features
print(dim_features.head())





# Drop the redundant 'features_id_y' column if it exists
propertyrecords_df = propertyrecords_df.drop(columns=['features_id_y'], errors='ignore')

# Rename 'features_id_x' to a clearer name
propertyrecords_df = propertyrecords_df.rename(columns={'features_id_x': 'features_id'})

# View the updated DataFrame
print(propertyrecords_df.head())




import json

# Columns to clean (excluding 'features' as you've handled that separately)
json_columns = ['taxAssessments', 'propertyTaxes', 'history', 'owner', 'hoa']

def safe_json_fix(x):
    if isinstance(x, str):
        try:
            # Try to parse it if it's double-encoded
            loaded = json.loads(x)
            # If it's still a string after first load, try once more
            if isinstance(loaded, str):
                loaded = json.loads(loaded)
            return json.dumps(loaded)
        except (json.JSONDecodeError, TypeError):
            return json.dumps(x)  # Wrap raw strings in JSON
    else:
        return json.dumps(x)  # Properly encode non-string types

# Apply to each JSON-like column
for col in json_columns:
    propertyrecords_df[col] = propertyrecords_df[col].apply(safe_json_fix)





# Merge the location table with the main DataFrame to get location_id
propertyrecords_df = propertyrecords_df.merge(dim_location_columns[['city', 'state', 'zipCode', 'county', 'location_id']],
                                              on=['city', 'state', 'zipCode', 'county'],
                                              how='left')

# Check if location_id is now present
print(propertyrecords_df[['location_id']].head())







# Merge the sales table with the main DataFrame to get sales_id
propertyrecords_df = propertyrecords_df.merge(
    dim_sale_columns[['lastSaleDate', 'lastSalePrice', 'sales_id']],
    on=['lastSaleDate', 'lastSalePrice'],
    how='left'
)

# Check if sales_id is now present
print(propertyrecords_df[['sales_id']].head())




fact_columns = [
    'id',
    'location_id',  # Foreign key from dim_location
    'sales_id',     # Foreign key from dim_sale
    'features_id',  # Foreign key from dim_features
    'bedrooms',
    'bathrooms',
    'squareFootage',
    'lotSize',
    'yearBuilt',
    'assessorID',
    'legalDescription',
    'subdivision',
    'zoning',
    'lastSaleDate',
    'lastSalePrice',
    'ownerOccupied',
    'hoa'
]

# Create the fact_table
fact_table = propertyrecords_df[fact_columns]

# View the fact_table
print(fact_table.head())

 




import os
import pandas as pd

# Save DataFrames if CSVs don't exist yet
def save_dataframe_to_csv(df, filename):
    if not os.path.exists(filename):
        df.to_csv(filename, index=False)
        print(f"[✓] File saved: {filename}")
    else:
        print(f"[✓] File already exists: {filename}")

# Load from CSV if exists, otherwise fallback to in-memory DataFrame
def load_or_fallback(filename, fallback_df):
    if os.path.exists(filename):
        print(f"[✓] Loaded: {filename}")
        return pd.read_csv(filename)
    else:
        print(f"[!] File not found: {filename}, using fallback data.")
        return fallback_df

# Save data
save_dataframe_to_csv(dim_features.reset_index(), 'dim_features.csv')
save_dataframe_to_csv(dim_location_columns, 'dim_location.csv')
save_dataframe_to_csv(dim_sale_columns, 'dim_sale.csv')
save_dataframe_to_csv(fact_table, 'fact_property.csv')

# Load or use existing DataFrames
dim_features_loaded = load_or_fallback('dim_features.csv', dim_features.reset_index())
dim_location_loaded = load_or_fallback('dim_location.csv', dim_location_columns)
dim_sale_loaded = load_or_fallback('dim_sale.csv', dim_sale_columns)
fact_table_loaded = load_or_fallback('fact_property.csv', fact_table)

# Preview
print(dim_features_loaded.head())



# Load the CSV file
df = pd.read_csv('dim_features.csv')

# Define function to parse JSON safely
def parse_json_column(x):
    if pd.isna(x):
        return None
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return None

# Apply it to the features column
df['features'] = df['features'].apply(parse_json_column)

# Optional: check the result
print(df.head())

dim_features.reset_index().to_csv('dim_features.csv', index=False)

df = pd.read_csv('dim_features.csv')


import pandas as pd
import json

# Load updated dim_features.csv
dim_features = pd.read_csv('dim_features.csv')
# Check for NaN values in the entire DataFrame
print(dim_features .isna().sum())




import pandas as pd

df = pd.read_csv('fact_property.csv')

# Drop 'hoa' column if it exists
if 'hoa' in df.columns:
    df = df.drop(columns=['hoa'])

# If fact_table is defined elsewhere and you want to drop 'hoa' from it too:
# fact_table = fact_table.drop(columns=['hoa'], errors='ignore')

# Save the cleaned DataFrame back to CSV
df.to_csv('fact_property.csv', index=False)

# Show current column names
print(df.columns.tolist())



import psycopg2

conn = psycopg2.connect(
   dbname="mydb",
    user="postgres",
    password="Oebuka2019",
    host="localhost",
    port="5432"
)

cur = conn.cursor()
cur.execute("SELECT version();")
print("Connected:", cur.fetchone())

cur.close()
conn.close()





import psycopg2

def create_table():
    # Establish a fresh connection
    conn = psycopg2.connect(
    dbname="mydb",
    user="postgres",
    password="Oebuka2019",
    host="localhost",
    port="5432"
    )
    cursor = conn.cursor()

    create_table_query = '''
    CREATE SCHEMA IF NOT EXISTS zapbank;

    DROP TABLE IF EXISTS zapbank.fact_table;
    DROP TABLE IF EXISTS zapbank.dim_features;
    DROP TABLE IF EXISTS zapbank.dim_location_columns;
    DROP TABLE IF EXISTS zapbank.dim_sale_columns;

    CREATE TABLE zapbank.dim_location_columns (
        location_id SERIAL PRIMARY KEY,
        city TEXT,
        state TEXT,
        zip_code TEXT,
        county TEXT
    );

    CREATE TABLE zapbank.dim_features (
        features_id SERIAL PRIMARY KEY,
        features TEXT
    );

    CREATE TABLE zapbank.dim_sale_columns (
        sales_id SERIAL PRIMARY KEY,
        level_0 NUMERIC,
        index NUMERIC,
        lastSaleDate DATE,
        lastSalePrice NUMERIC
    );

    CREATE TABLE zapbank.fact_table (
        id TEXT PRIMARY KEY,
        location_id INT REFERENCES zapbank.dim_location_columns(location_id),
        sales_id INT REFERENCES zapbank.dim_sale_columns(sales_id),
        features_id INT REFERENCES zapbank.dim_features(features_id),
        bedrooms INT,
        bathrooms INT,
        square_footage NUMERIC,
        lot_size NUMERIC,
        year_built INT,
        assessor_id TEXT,
        legal_description TEXT,
        subdivision TEXT,
        zoning TEXT,
        lastSaleDate DATE,
        lastSalePrice NUMERIC,
        ownerOccupied NUMERIC
    );
    '''

    cursor.execute(create_table_query)
    conn.commit()
    cursor.close()
    conn.close()
    print("Tables created successfully.")

# Now call the function
create_table()




import psycopg2

conn = psycopg2.connect(
dbname="mydb",
user="postgres",
password="Oebuka2019",
host="localhost",
port="5432"
)
cursor = conn.cursor()

cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'zapbank';")
print(cursor.fetchall())

cursor.close()
conn.close()





import os

for root, dirs, files in os.walk('/home/ochuko'):
    for file in files:
        if file.endswith('.csv'):
            print(os.path.join(root, file))







import psycopg2
import csv

# Function to establish a database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="mydb",
            user="postgres",
            password="Oebuka2019",
            host="localhost",
            port="5432"
        )
        return conn
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to clean and convert the data before insertion
def clean_row_data(row):
    clean_row = []
    for value in row:
        if value == '':  # Handle empty values (convert to None)
            clean_row.append(None)
        else:
            try:
                # Attempt to convert strings like "3.0" to integers
                if '.' in value:
                    clean_row.append(int(float(value)))
                else:
                    clean_row.append(int(value))  # Convert to integer
            except ValueError:
                clean_row.append(value)  # Leave as is if it can't be converted
    return clean_row

# Function to load data from CSV into dim_location_columns
def load_dim_location():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        with open('/home/ochuko/dim_location.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            
            # Insert data into the table
            for row in reader:
                # Clean the row data
                clean_row = clean_row_data(row)
                
                # Create placeholders for values
                placeholders = ','.join(['%s'] * len(clean_row))
                
                query = f'INSERT INTO zapbank.dim_location_columns VALUES ({placeholders}) ON CONFLICT DO NOTHING;'
                cursor.execute(query, clean_row)

        conn.commit()  # Commit changes to the database
        cursor.close()
        conn.close()
        print("Data loaded successfully into dim_location_columns")
    else:
        print("Failed to connect to the database.")

# Load the data into dim_location_columns
load_dim_location()






import psycopg2
import csv
from datetime import datetime

# Function to establish a database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="mydb",
            user="postgres",
            password="Oebuka2019",
            host="localhost",
            port="5432"
        )
        return conn
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to clean and convert the data before insertion
def clean_row_data(row):
    clean_row = []
    for value in row:
        if value == '' or value.lower() == 'unknown':
            clean_row.append(None)
        else:
            # Handle ISO date strings like 2022-05-03T00:00:00.000Z
            try:
                if 'T' in value and 'Z' in value:
                    clean_row.append(datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").date())
                else:
                    clean_row.append(value)
            except Exception:
                clean_row.append(value)
    return clean_row

# Function to load data from CSV into dim_sale_columns
def load_dim_sale():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        with open('/home/ochuko/dim_sale.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # Read the header row

            column_names = ', '.join(header)  # Convert list to comma-separated string

            for row in reader:
                clean_row = clean_row_data(row)
                placeholders = ','.join(['%s'] * len(clean_row))

                query = f'''
                    INSERT INTO zapbank.dim_sale_columns ({column_names})
                    VALUES ({placeholders})
                    ON CONFLICT DO NOTHING;
                '''
                cursor.execute(query, clean_row)

        conn.commit()
        cursor.close()
        conn.close()
        print("Data loaded successfully into dim_sale_columns")
    else:
        print("Failed to connect to the database.")

# Run the function
load_dim_sale()









import psycopg2
import csv

# Function to establish a database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="mydb",
            user="postgres",
            password="Oebuka2019",
            host="localhost",
            port="5432"
        )
        return conn
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to clean and convert the data before insertion
def clean_row_data(row):
    clean_row = []
    for value in row:
        if value == '':  # Handle empty values (convert to None)
            clean_row.append(None)
        else:
            try:
                # Attempt to convert strings like "3.0" to integers
                if '.' in value:
                    clean_row.append(int(float(value)))
                else:
                    clean_row.append(int(value))  # Convert to integer
            except ValueError:
                clean_row.append(value)  # Leave as is if it can't be converted
    return clean_row

# Function to load data from CSV into dim_features
def load_dim_features():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        with open('/home/ochuko/dim_features.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            
            # Insert data into the table
            for row in reader:
                # Clean the row data
                clean_row = clean_row_data(row)
                
                # Create placeholders for values
                placeholders = ','.join(['%s'] * len(clean_row))
                
                query = f'INSERT INTO zapbank.dim_features VALUES ({placeholders}) ON CONFLICT DO NOTHING;'
                cursor.execute(query, clean_row)

        conn.commit()  # Commit changes to the database
        cursor.close()
        conn.close()
        print("Data loaded successfully into dim_features")
    else:
        print("Failed to connect to the database.")

# Load the data into dim_features
load_dim_features()







import psycopg2
import csv
from datetime import datetime

# Function to establish a database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="mydb",
            user="postgres",
            password="Oebuka2019",
            host="localhost",
            port="5432"
        )
        return conn
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to clean and convert the data before insertion
def clean_row_data(row):
    clean_row = []
    for value in row:
        if value == '':  # Handle empty values (convert to None)
            clean_row.append(None)
        elif value == 'unknown':  # Handle 'unknown' for date columns
            clean_row.append(None)  # or you could replace it with a default date like '1970-01-01'
        else:
            try:
                # Attempt to convert strings like "3.0" to integers
                if '.' in value:
                    clean_row.append(int(float(value)))
                elif '/' in value:  # Handle date format like 'YYYY-MM-DD'
                    clean_row.append(value)  # Leave it as string, or convert using datetime if necessary
                else:
                    clean_row.append(int(value))  # Convert to integer
            except ValueError:
                try:
                    # If it's not an integer, try to convert to date if the column should have dates
                    clean_row.append(datetime.strptime(value, '%Y-%m-%d') if '/' in value else value)
                except ValueError:
                    clean_row.append(value)  # Leave as is if it can't be converted
    return clean_row

# Function to load data from CSV into fact_table
def load_fact_table():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        with open('/home/ochuko/fact_property.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            
            # Insert data into the table
            for row in reader:
                # Clean the row data
                clean_row = clean_row_data(row)

                # Handling the fact table references
                # We need to replace the sales_id, location_id, and features_id with their corresponding ids
                # Fetch corresponding IDs for the foreign keys from the dimension tables
                location_id = clean_row[1]  # Adjust index for location_id
                sales_id = clean_row[2]     # Adjust index for sales_id
                features_id = clean_row[3]  # Adjust index for features_id
                
                # Query to fetch the corresponding location_id from the dim_location_columns table
                cursor.execute(f"SELECT location_id FROM zapbank.dim_location_columns WHERE location_id = %s;", (location_id,))
                location_id_db = cursor.fetchone()
                clean_row[1] = location_id_db[0] if location_id_db else None  # Replace with db value

                # Query to fetch the corresponding sales_id from the dim_sale_columns table
                cursor.execute(f"SELECT sales_id FROM zapbank.dim_sale_columns WHERE sales_id = %s;", (sales_id,))
                sales_id_db = cursor.fetchone()
                clean_row[2] = sales_id_db[0] if sales_id_db else None  # Replace with db value

                # Query to fetch the corresponding features_id from the dim_features table
                cursor.execute(f"SELECT features_id FROM zapbank.dim_features WHERE features_id = %s;", (features_id,))
                features_id_db = cursor.fetchone()
                clean_row[3] = features_id_db[0] if features_id_db else None  # Replace with db value
                
                # Create placeholders for values
                placeholders = ','.join(['%s'] * len(clean_row))

                # Insert data into the fact_table
                query = f'INSERT INTO zapbank.fact_table VALUES ({placeholders}) ON CONFLICT DO NOTHING;'
                cursor.execute(query, clean_row)

        conn.commit()  # Commit changes to the database
        cursor.close()
        conn.close()
        print("Data loaded successfully into fact_table")
    else:
        print("Failed to connect to the database.")

# Load the data into fact_table
load_fact_table()





