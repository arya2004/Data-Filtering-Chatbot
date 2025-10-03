import pandas as pd
from openai import OpenAI
import re
import os
import time
from dotenv import load_dotenv
from openpyxl import load_workbook

import warnings
warnings.filterwarnings('ignore')

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PREDICTED_EXCEL_PATH = os.getenv("PREDICTED_EXCEL_PATH", "./predicted.xlsx")
INPUT_CSV_PATH = os.getenv("INPUT_CSV_PATH", "./input_table.csv")
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------- Utility Function -------------------
def load_input_csv(path):
    """
    Load a CSV file and parse the 'Date' column if it exists.
    Returns a pandas DataFrame.
    """
    try:
        df = pd.read_csv(path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        print(f"Error loading CSV file at {path}: {e}")
        return pd.DataFrame()

# ------------------- Dataset Summary -------------------
def get_dataset_summary(df):
    return df.describe(include='all').to_string()

# ------------------- Filtering Instructions -------------------
def get_filter_instructions(summary, question):
    extra_instructions = ""
    q_lower = question.lower()
    if "recent" in q_lower or "latest" in q_lower:
        extra_instructions += (
            "Note: When the question refers to 'recent' or 'latest entry', do not filter based on "
            "the date column. Instead, select the last row of the dataframe using its original order "
            "(e.g., use df.iloc[-1:]).\n"
        )
    if "early entry" in q_lower or "first entry" in q_lower or "earliest entry" in q_lower:
        extra_instructions += (
            "Note: When the question refers to 'early entry', 'first entry', or 'earliest entry', do not filter "
            "based on the date column. Instead, select the first row of the dataframe using its original order "
            "(e.g., use df.iloc[:1]).\n"
        )
    
    prompt = (
        "You are an assistant that helps filter a dataset based on a user query.\n"
        "You are given the dataset summary below:\n\n"
        f"{summary}\n\n"
        "And the following question:\n"
        f"\"{question}\"\n\n"
        f"{extra_instructions}"
        "Return only a Python code snippet that, when executed, filters the dataframe "
        "(named 'df') to only include the relevant rows and columns for this query. "
        "For example, if the answer required filtering rows where a column 'Age' > 30, "
        "your code might look like: df = df[df['Age'] > 30]\n\n"
        "Make sure to return only the code, optionally wrapped in triple backticks."
    )
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL,  
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.0
    )
    
    return response.choices[0].message.content

# ------------------- Final Answer -------------------
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
    
    return response.choices[0].message.content

# ------------------- Process Submission -------------------
def process_submission(excel_file_path, csv_data_path, dev_start=-1, dev_end=-1):
    df = load_input_csv(csv_data_path)
    if df.empty:
        return
    
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

        code_match = re.search(r"```(?:python)?\n(.*?)```", filter_instructions, re.DOTALL)
        code_to_execute = code_match.group(1) if code_match else filter_instructions

        try:
            local_vars = {"df": df_copy}
            exec(code_to_execute, {}, local_vars)
            filtered_df = local_vars.get("df", df_copy)

            if not isinstance(filtered_df, pd.DataFrame):
                print("Warning: Filtering code did not return a DataFrame. Using original dataset.")
                filtered_df = df_copy
            elif isinstance(filtered_df, pd.Series):
                filtered_df = filtered_df.to_frame().T
        except Exception as e:
            print("Error executing filtering code:", e)
            filtered_df = df_copy

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

# ------------------- Main Function -------------------
def main():
    df = load_input_csv(INPUT_CSV_PATH)
    if df.empty:
        return

    full_summary = get_dataset_summary(df)
    print("\nFull Dataset Summary:\n")
    print(full_summary)

    question = "Give me all row numbers where 10 quantities were sold"
    filter_instructions = get_filter_instructions(full_summary, question)
    print("\nLLM-provided filtering instructions:")
    print(filter_instructions)

    code_match = re.search(r"```(?:python)?\n(.*?)```", filter_instructions, re.DOTALL)
    code_to_execute = code_match.group(1) if code_match else filter_instructions

    try:
        local_vars = {"df": df}
        exec(code_to_execute, {}, local_vars)
        filtered_df = local_vars.get("df", df)
        if isinstance(filtered_df, pd.Series):
            filtered_df = filtered_df.to_frame().T
    except Exception as e:
        print("Error executing filtering code:", e)
        filtered_df = df

    filtered_summary = get_dataset_summary(filtered_df)
    print("\nFiltered Dataset Summary:\n")
    print(filtered_summary)

    final_answer = get_final_answer(filtered_df, filtered_summary, question)
    print("\nFinal Answer:\n")
    print(final_answer)

# ------------------- Entry Point -------------------
if __name__ == "__main__":
    # main()  # optional test run
    process_submission(PREDICTED_EXCEL_PATH, INPUT_CSV_PATH)
