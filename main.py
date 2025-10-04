import pandas as pd
from openai import OpenAI
import re
import os
import time
from dotenv import load_dotenv
from openpyxl import load_workbook
import json

import warnings
warnings.filterwarnings('ignore')


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PREDICTED_EXCEL_PATH = os.getenv("PREDICTED_EXCEL_PATH", "./predicted.xlsx")
INPUT_CSV_PATH = os.getenv("INPUT_CSV_PATH", "./input_table.csv")
client = OpenAI(api_key=OPENAI_API_KEY)

def detect_date_columns(df):
    date_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(date_word in col_lower for date_word in ['date', 'time', 'created', 'updated', 'timestamp']):
            try:
                df[col] = pd.to_datetime(df[col])
                date_columns.append(col)
            except (ValueError, TypeError):
                pass
        elif df[col].dtype == 'object':
            sample_values = df[col].dropna().head(3)
            if len(sample_values) > 0:
                try:
                    pd.to_datetime(sample_values.iloc[0])
                    df[col] = pd.to_datetime(df[col])
                    date_columns.append(col)
                except (ValueError, TypeError):
                    pass
    return df, date_columns

def get_dataset_summary(df):
    summary_parts = []
    
    summary_parts.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    summary_parts.append("\nColumn Schema:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        summary_parts.append(f"  - {col}: {dtype} (nulls: {null_count}, unique: {unique_count})")
        
        if dtype == 'object' or 'category' in dtype:
            sample_values = df[col].dropna().unique()[:3]
            summary_parts.append(f"    Sample values: {list(sample_values)}")
        elif df[col].dtype in ['int64', 'float64']:
            min_val = df[col].min()
            max_val = df[col].max()
            summary_parts.append(f"    Range: {min_val} to {max_val}")
    
    summary_parts.append("\nStatistical Summary:")
    summary_parts.append(df.describe(include='all').to_string())
    
    return "\n".join(summary_parts)

def get_filter_instructions(summary, question):
    prompt = (
        "You are an assistant that helps filter a dataset based on a user query.\n"
        "You have access to a dataset with the following schema and summary:\n\n"
        f"{summary}\n\n"
        "IMPORTANT: Use the exact column names shown in the schema above. "
        "Pay attention to data types for appropriate comparisons.\n\n"
        "Question to answer:\n"
        f"\"{question}\"\n\n"
        "Return a JSON object with the filter conditions. The JSON should have 'column', 'operator', and 'value'.\n"
        "For special cases like 'latest' or 'first' entry, return a JSON with a 'special' key.\n"
        "Examples:\n"
        '- For numeric: { "column": "ColumnName", "operator": ">", "value": 30 }\n'
        '- For string: { "column": "ColumnName", "operator": "contains", "value": "text" }\n'
        '- For date: { "column": "DateColumn", "operator": ">", "value": "2023-01-01" }\n'
        '- For latest entry: { "special": "latest" }\n'
        '- For first entry: { "special": "first" }\n\n'
        "Make sure to return only the JSON object, optionally wrapped in triple backticks."
    )
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL,  
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.0
    )
    
    instructions = response.choices[0].message.content
    return instructions

def get_final_answer(filtered_df, filtered_summary, question):
    prompt = (
        "You are an analytical assistant. Given the filtered dataset (its complete content and summary), "
        "answer the question using as few words as possible. Output only the essential answer in minimal words or numbers on separate lines, "
        "without any extra commentary. "
        "IMPORTANT: Ensure that each number or word is reproduced exactly as given in the table. "
        "Do not add timestamps, extra decimals, or any additional formatting. "
        "For example, do not change '2019-01-01' to '2019-01-01 00:00:00' or '5' to '5.0'.\n\n"
        "Filtered dataset (complete):\n"
        f"{filtered_df.to_string(index=False)}\n\n"
        "Filtered dataset summary:\n"
        f"{filtered_summary}\n\n"
        f"Question: \"{question}\"\n\n"
        "Please provide your answer in the following format: each answer on a new line with no additional text, exactly as in the table."
    )
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL, 
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.0
    )
    
    answer = response.choices[0].message.content
    return answer

def apply_filter(df, filter_json):
    try:
        # Attempt to clean and load the JSON
        # It might be wrapped in backticks or have trailing characters
        json_match = re.search(r'\{.*\}', filter_json, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0)
            filter_data = json.loads(clean_json)
        else:
            print("Error: No valid JSON object found in the string from LLM")
            return df
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON from LLM after cleaning: {e}")
        return df

    if "special" in filter_data:
        if filter_data["special"] == "latest":
            return df.iloc[-1:]
        elif filter_data["special"] == "first":
            return df.iloc[:1]
    
    column = filter_data.get("column")
    operator = filter_data.get("operator")
    value = filter_data.get("value")

    if not all([column, operator, value is not None]):
        print("Error: Incomplete filter from LLM")
        return df

    if column not in df.columns:
        print(f"Error: Column '{column}' not found in DataFrame")
        return df

    try:
        if operator == "contains":
            return df[df[column].str.contains(str(value), case=False, na=False)]
        else:
            # For numeric and date comparisons, pandas.query is safer
            if pd.api.types.is_numeric_dtype(df[column]):
                query_str = f"`{column}` {operator} {value}"
            else: # Assume string or date, wrap value in quotes
                query_str = f"`{column}` {operator} '{value}'"
            return df.query(query_str)
    except Exception as e:
        print(f"Error applying filter: {e}")
        return df

def process_submission(excel_file_path, csv_data_path, dev_start=-1, dev_end=-1):

 
    try:
        df = pd.read_csv(csv_data_path)
    except Exception as e:
        print("Error loading CSV file:", e)
        return
    
    df, date_columns = detect_date_columns(df)
    
    full_summary = get_dataset_summary(df)


    if os.path.exists(excel_file_path):
        try:
            submission_df = pd.read_excel(excel_file_path)
        except Exception as e:
            print("Error loading Excel file:", e)
            return
    else:
        print("Excel file not found. Creating a new submission DataFrame.")
        submission_df = pd.DataFrame(columns=["question", "row index", "column index", "answer"])


    for col in ["filtered row index", "filtered column index", "generated response"]:
        if col not in submission_df.columns:
            submission_df[col] = ""


    if dev_start == -1 and dev_end == -1:
        indices_to_process = submission_df.index.tolist()
    else:
        indices_to_process = [i for i in submission_df.index if dev_start <= i <= dev_end]

    for idx in indices_to_process:
        row = submission_df.loc[idx]
        question = row["question"]
        print(f"Processing question at index {idx}: {question}")

    
        df_copy = df.copy()


        filter_instructions = get_filter_instructions(full_summary, question)
        print("LLM-provided filtering instructions:")
        print(filter_instructions)

        # Clean up the JSON string
        json_match = re.search(r"```(?:json)?\n(.*?)```", filter_instructions, re.DOTALL)
        if json_match:
            filter_json = json_match.group(1)
        else:
            filter_json = filter_instructions

        filtered_df = apply_filter(df_copy, filter_json)

        filtered_row_indices = list(filtered_df.index)
        filtered_column_indices = [df.columns.get_loc(col) for col in filtered_df.columns if col in df.columns]

        filtered_summary = get_dataset_summary(filtered_df)

   
        generated_response = get_final_answer(filtered_df, filtered_summary, question)

    
        submission_df.at[idx, "filtered row index"] = ", ".join(map(str, filtered_row_indices))
        submission_df.at[idx, "filtered column index"] = ", ".join(map(str, filtered_column_indices))
        submission_df.at[idx, "generated response"] = generated_response

  
        success = False
        retries = 3
        while not success and retries > 0:
            try:
                with pd.ExcelWriter(excel_file_path, engine="openpyxl", mode="w") as writer:
                    submission_df.to_excel(writer, index=False)
                success = True
                print(f"Updated Excel file at {excel_file_path} after processing index {idx}.")
            except Exception as e:
                print(f"Error saving Excel file: {e}")
                retries -= 1
                time.sleep(2)  
                if retries == 0:
                    print(f"Failed to save after multiple attempts. Skipping update for index {idx}.")

    print("Submission processing complete.")

def main():

    csv_path = INPUT_CSV_PATH
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print("Error loading CSV file:", e)
        return
    
    df, date_columns = detect_date_columns(df)
    
    full_summary = get_dataset_summary(df)
    print("\nFull Dataset Summary:\n")
    print(full_summary)

    question = "Give me all row numbers where 10 quantities were sold"


    filter_instructions = get_filter_instructions(full_summary, question)
    print("\nLLM-provided filtering instructions:")
    print(filter_instructions)

    filtered_df = apply_filter(df, filter_instructions)

    filtered_summary = get_dataset_summary(filtered_df)
    print("\nFiltered Dataset Summary:\n")
    print(filtered_summary)

    final_answer = get_final_answer(filtered_df, filtered_summary, question)
    print("\nFinal Answer:\n")
    print(final_answer)

if __name__ == "__main__":

    # main()

    excel_file_path = PREDICTED_EXCEL_PATH  
    csv_data_path = INPUT_CSV_PATH     
    

    dev_start = -1
    dev_end = -1
    
    process_submission(excel_file_path, csv_data_path, dev_start, dev_end)
