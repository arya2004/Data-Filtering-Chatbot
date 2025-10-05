import pandas as pd
from openai import OpenAI
import re
import os
import time
import json
from dotenv import load_dotenv
from openpyxl import load_workbook
import operator
import warnings

warnings.filterwarnings('ignore')

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PREDICTED_EXCEL_PATH = os.getenv("PREDICTED_EXCEL_PATH", "./predicted.xlsx")
INPUT_CSV_PATH = os.getenv("INPUT_CSV_PATH", "./input_table.csv")
client = OpenAI(api_key=OPENAI_API_KEY)


# ------------------- State Management -------------------
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


# ------------------- Utility Functions -------------------
def load_input_csv(path):
    """Load a CSV file and parse the 'Date' column if it exists."""
    try:
        df = pd.read_csv(path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        print(f"Error loading CSV file at {path}: {e}")
        return pd.DataFrame()


def detect_date_columns(df):
    date_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in ['date', 'time', 'created', 'updated', 'timestamp']):
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
    summary_parts = [f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns", "\nColumn Schema:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        summary_parts.append(f"  - {col}: {dtype} (nulls: {null_count}, unique: {unique_count})")

        if dtype == 'object' or 'category' in dtype:
            sample_values = df[col].dropna().unique()[:3]
            summary_parts.append(f"    Sample values: {list(sample_values)}")
        elif df[col].dtype in ['int64', 'float64']:
            summary_parts.append(f"    Range: {df[col].min()} to {df[col].max()}")

    summary_parts.append("\nStatistical Summary:")
    summary_parts.append(df.describe(include='all').to_string())
    return "\n".join(summary_parts)


# ------------------- Filtering Instructions -------------------
def get_filter_instructions(summary, question):
    extra_instructions = ""
    q_lower = question.lower()
    if "recent" in q_lower or "latest" in q_lower:
        extra_instructions += (
            "Note: For 'recent' or 'latest entry', select the last row (df.iloc[-1:]).\n"
        )
    if "early entry" in q_lower or "first entry" in q_lower or "earliest entry" in q_lower:
        extra_instructions += (
            "Note: For 'early' or 'first entry', select the first row (df.iloc[:1]).\n"
        )

    prompt = (
        "You are an assistant that helps filter a dataset based on a user query.\n"
        f"{summary}\n\n"
        f"Question: \"{question}\"\n\n"
        f"{extra_instructions}"
        "Return only JSON with 'column', 'operator', and 'value'. "
        "Supported operators: >, <, ==, !=, >=, <=, contains."
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.0,
    )
    return response.choices[0].message.content


# ------------------- Final Answer -------------------
def get_final_answer(filtered_df, filtered_summary, question):
    prompt = (
        "You are an analytical assistant. Given the filtered dataset and summary, "
        "answer the question concisely with exact values.\n\n"
        f"{filtered_df.to_string(index=False)}\n\n"
        f"{filtered_summary}\n\n"
        f"Question: \"{question}\""
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.0,
    )
    return response.choices[0].message.content


# ------------------- Process Submission -------------------
def process_submission(excel_file_path, csv_data_path, dev_start=-1, dev_end=-1):
    df = load_input_csv(csv_data_path)
    if df.empty:
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

    state_file_path = "state.json"
    results_file_path = "intermediate_results.json"
    processing_state = load_processing_state(state_file_path)
    completed_indices = set(processing_state.get("completed_indices", []))

    if dev_start == -1 and dev_end == -1:
        indices_to_process = submission_df.index.tolist()
    else:
        indices_to_process = [i for i in submission_df.index if dev_start <= i <= dev_end]

    remaining_indices = [idx for idx in indices_to_process if idx not in completed_indices]

    if completed_indices:
        print(f"Resuming processing: {len(completed_indices)} done, {len(remaining_indices)} remaining")

    try:
        for idx in remaining_indices:
            row = submission_df.loc[idx]
            question = row["question"]
            print(f"\nProcessing question {idx}: {question}")

            df_copy = df.copy()
            filter_instructions = get_filter_instructions(full_summary, question)
            json_match = re.search(r"```(?:json)?\n(.*?)```", filter_instructions, re.DOTALL)
            filter_str = json_match.group(1) if json_match else filter_instructions

            try:
                filter_json = json.loads(filter_str)
                col = filter_json['column']
                op = filter_json['operator']
                val = filter_json['value']

                OPS = {
                    ">": operator.gt, "<": operator.lt, ">=": operator.ge,
                    "<=": operator.le, "==": operator.eq, "!=": operator.ne
                }

                try:
                    if op == "contains":
                        filtered_df = df_copy[df_copy[col].astype(str).str.contains(str(val), case=False, na=False)]
                    elif op in OPS:
                        func = OPS[op]
                        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                            val = pd.to_datetime(val)
                        filtered_df = df_copy[func(df_copy[col], val)]
                    else:
                        print(f"Unsupported operator '{op}'. Using full dataset.")
                        filtered_df = df_copy
                except Exception as e:
                    print(f"Error applying safe filter: {e}")
                    filtered_df = df_copy

            except (json.JSONDecodeError, KeyError, Exception) as e:
                print(f"Error parsing filter JSON: {e}. Using full dataset.")
                filtered_df = df_copy

            if not isinstance(filtered_df, pd.DataFrame):
                filtered_df = df_copy

            filtered_row_indices = list(filtered_df.index)
            filtered_column_indices = [df.columns.get_loc(c) for c in filtered_df.columns if c in df.columns]

            filtered_summary = get_dataset_summary(filtered_df)
            generated_response = get_final_answer(filtered_df, filtered_summary, question)

            result_data = {
                "filtered_row_index": ", ".join(map(str, filtered_row_indices)),
                "filtered_column_index": ", ".join(map(str, filtered_column_indices)),
                "generated_response": generated_response,
            }

            save_intermediate_result(results_file_path, idx, result_data)
            submission_df.at[idx, "filtered row index"] = result_data["filtered_row_index"]
            submission_df.at[idx, "filtered column index"] = result_data["filtered_column_index"]
            submission_df.at[idx, "generated response"] = result_data["generated_response"]

            completed_indices.add(idx)
            processing_state["completed_indices"] = list(completed_indices)
            save_processing_state(state_file_path, processing_state)

            print(f"Completed index {idx}")

    except Exception as e:
        print(f"Processing interrupted: {e}")
        print("State saved for resume.")
        return

    print("All processing complete. Building Excel output...")
    try:
        with pd.ExcelWriter(excel_file_path, engine="openpyxl", mode="w") as writer:
            submission_df.to_excel(writer, index=False)
        print(f"Final Excel saved: {excel_file_path}")
    except Exception as e:
        print(f"Error saving Excel: {e}")

    for f in ["state.json", "intermediate_results.json"]:
        if os.path.exists(f):
            os.remove(f)
    print("Processing state files cleaned up.\nSubmission processing complete.")


# ------------------- Main -------------------
def main():
    df = load_input_csv(INPUT_CSV_PATH)
    if df.empty:
        return

    df, date_columns = detect_date_columns(df)
    full_summary = get_dataset_summary(df)

    question = "Give me all row numbers where 10 quantities were sold"
    filter_instructions = get_filter_instructions(full_summary, question)
    json_match = re.search(r"```(?:json)?\n(.*?)```", filter_instructions, re.DOTALL)
    filter_str = json_match.group(1) if json_match else filter_instructions

    try:
        filter_json = json.loads(filter_str)
        col = filter_json['column']
        op = filter_json['operator']
        val = filter_json['value']

        OPS = {
            ">": operator.gt, "<": operator.lt, ">=": operator.ge,
            "<=": operator.le, "==": operator.eq, "!=": operator.ne
        }

        if op == "contains":
            filtered_df = df[df[col].astype(str).str.contains(str(val), case=False, na=False)]
        elif op in OPS:
            func = OPS[op]
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                val = pd.to_datetime(val)
            filtered_df = df[func(df[col], val)]
        else:
            print(f"Unsupported operator '{op}'. Using full dataset.")
            filtered_df = df

    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error applying filter: {e}. Using full dataset.")
        filtered_df = df

    filtered_summary = get_dataset_summary(filtered_df)
    print("\nFiltered Dataset Summary:\n")
    print(filtered_summary)

    final_answer = get_final_answer(filtered_df, filtered_summary, question)
    print("\nFinal Answer:\n")
    print(final_answer)


if __name__ == "__main__":
    process_submission(PREDICTED_EXCEL_PATH, INPUT_CSV_PATH)
