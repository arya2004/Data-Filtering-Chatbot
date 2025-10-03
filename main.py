import pandas as pd
from openai import OpenAI
import re
import os
import time
import json
from dotenv import load_dotenv
from openpyxl import load_workbook

import warnings
warnings.filterwarnings('ignore')


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def load_processing_state(state_file_path):
    """Load processing state from checkpoint file."""
    if os.path.exists(state_file_path):
        try:
            with open(state_file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"completed_indices": [], "results": {}}
    return {"completed_indices": [], "results": {}}

def save_processing_state(state_file_path, state):
    """Save processing state to checkpoint file."""
    try:
        with open(state_file_path, 'w') as f:
            json.dump(state, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save state checkpoint: {e}")

def save_intermediate_result(results_file_path, idx, result_data):
    """Save intermediate result to temporary file."""
    try:
        results = {}
        if os.path.exists(results_file_path):
            with open(results_file_path, 'r') as f:
                results = json.load(f)
        
        results[str(idx)] = result_data
        
        with open(results_file_path, 'w') as f:
            json.dump(results, f, indent=2)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not save intermediate result: {e}")

def get_dataset_summary(df):
    return df.describe(include='all').to_string()

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
        model="gpt-4o-mini",  
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
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.0
    )
    
    answer = response.choices[0].message.content
    return answer

def process_submission(excel_file_path, csv_data_path, dev_start=-1, dev_end=-1):
    # Setup state management files
    base_name = os.path.splitext(excel_file_path)[0]
    state_file_path = f"{base_name}_processing_state.json"
    results_file_path = f"{base_name}_intermediate_results.json"
    
    try:
        df = pd.read_csv(csv_data_path)
    except Exception as e:
        print("Error loading CSV file:", e)
        return
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
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

    # Load processing state for resume capability
    processing_state = load_processing_state(state_file_path)
    completed_indices = set(processing_state.get("completed_indices", []))
    
    if dev_start == -1 and dev_end == -1:
        indices_to_process = submission_df.index.tolist()
    else:
        indices_to_process = [i for i in submission_df.index if dev_start <= i <= dev_end]
    
    # Filter out already completed indices for resumable processing
    remaining_indices = [idx for idx in indices_to_process if idx not in completed_indices]
    
    if completed_indices:
        print(f"Resuming processing. Already completed: {len(completed_indices)} rows, Remaining: {len(remaining_indices)} rows")
    
    try:
        for idx in remaining_indices:
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

            # Save intermediate result before updating main dataframe
            result_data = {
                "filtered_row_index": ", ".join(map(str, filtered_row_indices)),
                "filtered_column_index": ", ".join(map(str, filtered_column_indices)),
                "generated_response": generated_response
            }
            save_intermediate_result(results_file_path, idx, result_data)

            # Update submission dataframe
            submission_df.at[idx, "filtered row index"] = result_data["filtered_row_index"]
            submission_df.at[idx, "filtered column index"] = result_data["filtered_column_index"]
            submission_df.at[idx, "generated response"] = result_data["generated_response"]

            # Update processing state checkpoint
            completed_indices.add(idx)
            processing_state["completed_indices"] = list(completed_indices)
            save_processing_state(state_file_path, processing_state)
            
            print(f"Completed processing for index {idx}")

    except Exception as e:
        print(f"Processing interrupted: {e}")
        print(f"State saved. You can resume by running the script again.")
        return

    # Build final Excel output only after all processing is complete
    print("All processing complete. Building final Excel output...")
    success = False
    retries = 3
    while not success and retries > 0:
        try:
            with pd.ExcelWriter(excel_file_path, engine="openpyxl", mode="w") as writer:
                submission_df.to_excel(writer, index=False)
            success = True
            print(f"Final Excel file saved: {excel_file_path}")
        except Exception as e:
            print(f"Error saving final Excel file: {e}")
            retries -= 1
            time.sleep(2)
            if retries == 0:
                print(f"Failed to save final Excel file after multiple attempts.")
                return

    # Clean up state files after successful completion
    try:
        if os.path.exists(state_file_path):
            os.remove(state_file_path)
        if os.path.exists(results_file_path):
            os.remove(results_file_path)
        print("Processing state files cleaned up.")
    except OSError:
        print("Warning: Could not clean up temporary state files.")

    print("Submission processing complete.")

def main():

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

if __name__ == "__main__":

    # main()

    excel_file_path = "./predicted.xlsx"  
    csv_data_path = "./input_table.csv"     
    

    dev_start = -1
    dev_end = -1
    
    process_submission(excel_file_path, csv_data_path, dev_start, dev_end)
