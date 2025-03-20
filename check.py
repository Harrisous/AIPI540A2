import pandas as pd
import json
from tqdm import tqdm

# Read processed data
df = pd.read_csv("models/data/processed_data.csv")

# Add new column to mark if requirements are met
df['is_valid_reconstruction'] = False

# Check each row
for index, row in tqdm(df.iterrows(), total=len(df)):
    original_comment = row["content"]
    
    # Skip rows without sub-comments
    if pd.isna(row["sub_comments"]):
        continue
    
    try:
        # Parse sub-comments JSON
        sub_comments_data = json.loads(row["sub_comments"])
        
        # Extract and concatenate all sub-comment texts
        reconstructed_comment = "".join([item["sub_comment"] for item in sub_comments_data])
        
        # Check if reconstructed comment matches the original
        is_valid = reconstructed_comment == original_comment
        
        # Update result column
        df.at[index, 'is_valid_reconstruction'] = is_valid
        
    except Exception as e:
        print(f"Error processing row {index}: {e}")

# Summarize results
valid_count = df['is_valid_reconstruction'].sum()
total_count = df['sub_comments'].notna().sum()
percentage = (valid_count / total_count * 100) if total_count > 0 else 0

print(f"\nCheck results:")
print(f"Total valid rows: {total_count}")
print(f"Correctly reconstructed: {valid_count}")
print(f"Accuracy: {percentage:.2f}%")

# Save results
df.to_csv("models/data/processed_data_checked.csv", index=False)
print("Results saved to models/data/processed_data_checked.csv")
