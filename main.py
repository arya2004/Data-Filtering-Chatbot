import pandas as pd
from openai import OpenAI
import re
import os
import time
from dotenv import load_dotenv
from openpyxl import load_workbook

import warnings
warnings.filterwarnings('ignore')

# Load environment variables and initialize OpenAI client.
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_dataset_summary(df):
    """Return a string summary of the dataframe."""
    return df.describe(include='all').to_string()

def get_filter_instructions(summary, question):
    """
    Ask the LLM to provide a Python code snippet that filters the dataframe.
    If the query mentions "recent" or "latest entry", instruct it to return the last row
    (using df.iloc[-1:]). Similarly, if it mentions "early entry" or "first entry", return
    the first row (using df.iloc[:1]).
    """
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
        model="gpt-4o-mini",  
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.0
    )
    
    instructions = response.choices[0].message.content
    return instructions

def get_final_answer(filtered_df, filtered_summary, question):
    """
    Ask the LLM for the final answer based on the filtered dataframe and its summary.
    The filtered dataframe should include only the relevant rows and columns.
    """
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
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.0
    )
    
    answer = response.choices[0].message.content
    return answer

def process_submission(excel_file_path, csv_data_path, dev_start=-1, dev_end=-1):
    """
    Process submission questions in an Excel file and save after each entry is appended.
    """
    # Load the original CSV dataset.
    try:
        df = pd.read_csv(csv_data_path)
    except Exception as e:
        print("Error loading CSV file:", e)
        return
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    full_summary = get_dataset_summary(df)

    # Load the Excel submission file; if it doesn't exist, create a new DataFrame.
    if os.path.exists(excel_file_path):
        try:
            submission_df = pd.read_excel(excel_file_path)
        except Exception as e:
            print("Error loading Excel file:", e)
            return
    else:
        print("Excel file not found. Creating a new submission DataFrame.")
        submission_df = pd.DataFrame(columns=["question", "row index", "column index", "answer"])

    # Ensure additional columns exist.
    for col in ["filtered row index", "filtered column index", "generated response"]:
        if col not in submission_df.columns:
            submission_df[col] = ""

    # Determine which rows to process.
    if dev_start == -1 and dev_end == -1:
        indices_to_process = submission_df.index.tolist()
    else:
        indices_to_process = [i for i in submission_df.index if dev_start <= i <= dev_end]

    for idx in indices_to_process:
        row = submission_df.loc[idx]
        question = row["question"]
        print(f"Processing question at index {idx}: {question}")

        # Use a fresh copy of the original dataset.
        df_copy = df.copy()

        # Get filtering instructions from the LLM.
        filter_instructions = get_filter_instructions(full_summary, question)
        print("LLM-provided filtering instructions:")
        print(filter_instructions)

        # Extract the Python code snippet (if wrapped in triple backticks).
        code_match = re.search(r"```(?:python)?\n(.*?)```", filter_instructions, re.DOTALL)
        code_to_execute = code_match.group(1) if code_match else filter_instructions

        # Execute the filtering code.
                # Execute the filtering code.
        try:
            local_vars = {"df": df_copy}
            exec(code_to_execute, {}, local_vars)
            filtered_df = local_vars.get("df", df_copy)
            # If the result is not a DataFrame, warn and fallback to the original dataset.
            if not isinstance(filtered_df, pd.DataFrame):
                print("Warning: Filtering code did not return a DataFrame. Using original dataset.")
                filtered_df = df_copy
            # If it’s a Series, convert it to a DataFrame.
            elif isinstance(filtered_df, pd.Series):
                filtered_df = filtered_df.to_frame().T
        except Exception as e:
            print("Error executing filtering code:", e)
            filtered_df = df_copy


        # Get filtered row and column indices.
        filtered_row_indices = list(filtered_df.index)
        filtered_column_indices = [df.columns.get_loc(col) for col in filtered_df.columns if col in df.columns]

        # Get summary of the filtered dataset.
        filtered_summary = get_dataset_summary(filtered_df)

        # Get final response from the LLM using the filtered dataframe and its summary.
        generated_response = get_final_answer(filtered_df, filtered_summary, question)

        # Update the submission DataFrame.
        submission_df.at[idx, "filtered row index"] = ", ".join(map(str, filtered_row_indices))
        submission_df.at[idx, "filtered column index"] = ", ".join(map(str, filtered_column_indices))
        submission_df.at[idx, "generated response"] = generated_response

        # Save the Excel file after each update.
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
                time.sleep(2)  # Wait before retrying
                if retries == 0:
                    print(f"Failed to save after multiple attempts. Skipping update for index {idx}.")

    print("Submission processing complete.")

def main():
    # Example main function demonstrating the original CSV filtering process.
    csv_path = "./input_table.csv"
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print("Error loading CSV file:", e)
        return
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    full_summary = get_dataset_summary(df)
    print("\nFull Dataset Summary:\n")
    print(full_summary)

    question = "Give me all row numbers where 10 quantities were sold"

    # Get filtering instructions.
    filter_instructions = get_filter_instructions(full_summary, question)
    print("\nLLM-provided filtering instructions:")
    print(filter_instructions)

    # Extract code snippet.
    code_match = re.search(r"```(?:python)?\n(.*?)```", filter_instructions, re.DOTALL)
    code_to_execute = code_match.group(1) if code_match else filter_instructions

    # Execute the filtering code.
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

if __name__ == "__main__":
    # Uncomment the following line to run the original CSV chatbot:
    # main()
    
    # Process submission: input and output file are the same.
    excel_file_path = "./predicted.xlsx"  # Input Excel file (also used for output)
    csv_data_path = "./input_table.csv"     # CSV file containing the dataset
    
    # Set developer start and end indices. If both are -1, process the entire sheet.
    dev_start = -1
    dev_end = -1
    
    process_submission(excel_file_path, csv_data_path, dev_start, dev_end)
