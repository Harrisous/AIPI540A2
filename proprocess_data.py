import json
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from llm_api import llm_generate

# Intermediate file paths
TEMP_FILE = "models/data/processing_temp.csv"
OUTPUT_FILE = "models/data/processed_data.csv"
# Number of threads for parallel processing
NUM_WORKERS = 20


# Define function to process a single record
@retry(
    stop=stop_after_attempt(3),  # Maximum 3 retry attempts
    wait=wait_exponential(
        multiplier=1, min=2, max=10
    ),  # Exponential backoff, 2-10 seconds
    retry=retry_if_exception_type(
        (ConnectionError, TimeoutError)
    ),  # Only retry network-related errors
    reraise=True,  # Re-raise exception on final failure
    before_sleep=lambda retry_state: print(
        f"Retrying {retry_state.attempt_number}/3 record {retry_state.args[0][0]}, waiting {retry_state.next_action.sleep} seconds..."
    ),
)
def process_single_record(record):
    index, content = record
    try:
        sub_comments = llm_generate(content)
        serialized = [user.model_dump() for user in sub_comments]
        json_str = json.dumps(serialized, ensure_ascii=False)
        return index, json_str
    except Exception as e:
        print(f"Failed to process record {index}: {e}")
        return index, None


# Check if intermediate file exists
if os.path.exists(TEMP_FILE):
    print(f"Found intermediate file, continuing processing...")
    df = pd.read_csv(TEMP_FILE)
    # Calculate number of processed records
    processed = df["sub_comments"].notna().sum()
    print(f"Processed {processed}/{len(df)} records")
else:
    print("Starting processing from scratch...")
    df = pd.read_csv(
        "models/data/sentiment-analysis-dataset-google-play-app-reviews.csv"
    )
    # Create new column
    df["sub_comments"] = None
    # Save initial state
    df.to_csv(TEMP_FILE, index=False)

try:
    # Get unprocessed records
    unprocessed_df = df[df["sub_comments"].isna()]

    if len(unprocessed_df) > 0:
        print(
            f"Starting parallel processing of {len(unprocessed_df)} records using {NUM_WORKERS} worker threads..."
        )
        # Prepare task list
        tasks = [(index, row["content"]) for index, row in unprocessed_df.iterrows()]

        # Create progress bar
        pbar = tqdm(total=len(tasks))

        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_record, task): task[0] for task in tasks
            }

            # Process completed tasks
            for future in as_completed(future_to_index):
                index, result = future.result()
                if result is not None:
                    df.at[index, "sub_comments"] = result

                # Update progress bar
                pbar.update(1)

                # Save progress after each record completion
                if pbar.n % 5 == 0 or pbar.n == len(
                    tasks
                ):  # Save every 5 records to reduce I/O operations
                    df.to_csv(TEMP_FILE, index=False)

                # Add a short delay to prevent API request throttling
                # time.sleep(0.1)

            pbar.close()

    # After all processing is complete, save to final file and delete intermediate file
    df.to_csv(OUTPUT_FILE, index=False)
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)
    print("Processing complete! Intermediate file deleted.")

except Exception as e:
    # Save current progress when exception occurs
    df.to_csv(TEMP_FILE, index=False)
    print(f"Processing interrupted: {e}")
    print(f"Progress saved to {TEMP_FILE}, will continue from checkpoint next time.")
