# 🛠️ Data Pipeline Project (ELT with Python & PostgreSQL)

## 🚀 Overview
This project showcases a practical ELT (Extract, Load, Transform) data pipeline built using **Python** in a **Jupyter Notebook** environment. It simulates a real-world scenario of ingesting raw property data, transforming it with pandas, and loading it into a PostgreSQL database for analysis.

## 🧰 Tools & Technologies
- Python 3  
- Jupyter Notebook  
- pandas  
- psycopg2  
- PostgreSQL  
- SQL  

## 🔁 Pipeline Workflow
1. **Extract** – Read raw property data from CSV files using pandas  
2. **Load** – Insert the raw data into PostgreSQL staging tables using psycopg2  
3. **Transform** – Clean, normalize, and structure data into star schema format (dimension & fact tables)

## 📂 Project Structure
- `pipeline_notebook.ipynb` – Main Jupyter Notebook with the full pipeline  
- `config.py` – PostgreSQL database connection settings  
- `data/` – Folder containing raw CSV source files  
- `requirements.txt` – List of Python dependencies  
- `README.md` – Project documentation (this file)

## ⚙️ Setup & Run

Install dependencies:
```bash
pip install -r requirements.txt

Launch Jupyter Notebook:
jupyter notebook



Open pipeline_notebook.ipynb and follow the steps inside.
Note: Update database credentials in config.py before running the connection cells.

🎯 Key Learning Outcomes
Using Jupyter Notebook to develop and explain ELT pipelines

Extracting data with pandas from raw CSV sources

Loading and managing data with PostgreSQL and psycopg2

Designing star schema with fact and dimension tables

Building scalable and reproducible data pipelines
