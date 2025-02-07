import pandas as pd
from openai import OpenAI
import re
import os
# Set your OpenAI API key (replace with your own key or load from environment)

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY,
)


def get_dataset_summary(df):
    """
    Return a string summary of the dataframe.
    You might use df.describe() for numeric columns and additional info if needed.
    """
    # Using describe for a quick summary; you can customize as needed.
    return df.describe(include='all').to_string()

def get_filter_instructions(summary, question):
    """
    Ask the LLM to provide a Python code snippet that filters the dataframe (variable 'df')
    based on the dataset summary and the question.
    """
    prompt = (
        "You are an assistant that helps filter a dataset based on a user query. \n"
        "You are given the dataset summary below:\n\n"
        f"{summary}\n\n"
        "And the following question:\n"
        f"\"{question}\"\n\n"
        "Return only a Python code snippet that, when executed, filters the dataframe "
        "(named 'df') to only include the relevant rows and columns for this query. "
        "For example, if the answer required filtering rows where a column 'Age' > 30, "
        "your code might look like: df = df[df['Age'] > 30]\n\n"
        "Make sure to return only the code, optionally wrapped in triple backticks."
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with the appropriate model name if different
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.0
    )
    
    instructions = response.choices[0].message.content
    return instructions

def get_final_answer(summary, filtered_summary, question):
    """
    Ask the LLM to provide an answer to the question based on the full dataset summary and
    the summary of the filtered data.
    """
    prompt = (
        "You are an analytical assistant. Given the full dataset summary and the filtered dataset summary, answer the question.\n\n"
        "Full dataset summary:\n"
        f"{summary}\n\n"
        "Filtered dataset summary:\n"
        f"{filtered_summary}\n\n"
        f"Question: \"{question}\"\n\n"
        "Provide a clear, concise answer."
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with the appropriate model name if different
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.0
    )
    
    answer =response.choices[0].message.content
    return answer

def main():
    # Ask the user for the CSV file path.
    csv_path = "./input_table.csv"
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print("Error loading CSV file:", e)
        return
    
    df['Date'] = pd.to_datetime(df['Date'])
    # Create a summary of the full dataset.
    full_summary = get_dataset_summary(df)
    print("\nFull Dataset Summary:\n")
    print(full_summary)

    # Get the user's question about the dataset.
    question = "What is the cost in the recent entry? "

    # Step 1: Ask the LLM for filtering instructions.
    filter_instructions = get_filter_instructions(full_summary, question)
    print("\nLLM-provided filtering instructions:")
    print(filter_instructions)

    # Extract code from the instructions if wrapped in triple backticks.
    code_match = re.search(r"```(?:python)?\n(.*?)```", filter_instructions, re.DOTALL)
    if code_match:
        code_to_execute = code_match.group(1)
    else:
        code_to_execute = filter_instructions  # assume the raw output is code

    # Step 2: Execute the filtering code.
    # WARNING: Executing untrusted code is dangerous. This is for demonstration only.
    try:
        local_vars = {"df": df}  # Provide the current dataframe in the local namespace.
        exec(code_to_execute, {}, local_vars)
        # Expect that the filtering code updates 'df'
        filtered_df = local_vars.get("df", df)
    except Exception as e:
        print("Error executing filtering code:", e)
        print("Proceeding with the original dataset.")
        filtered_df = df

    # Get a summary of the filtered dataset.
    filtered_summary = get_dataset_summary(filtered_df)
    print("\nFiltered Dataset Summary:\n")
    print(filtered_summary)

    # Step 3: Ask the LLM for the final answer using both summaries.
    final_answer = get_final_answer(full_summary, filtered_summary, question)
    print("\nFinal Answer:\n")
    print(final_answer)

if __name__ == "__main__":
    main()
