import os
import pandas as pd
from datasets import load_dataset

# Get the API token from the environment variable
api_token = os.getenv("API_TOKEN")

# Debugging: Print the API token to ensure it's being read correctly
print(f"API_TOKEN: {api_token}")

if not api_token:
    raise ValueError("No API token provided. Please set the API_TOKEN environment variable.")

# Set the Hugging Face API token for datasets library
os.environ["HF_API_TOKEN"] = api_token

# Load the dataset
dataset = load_dataset('amazon_polarity')

# Check the structure of the dataset
print(dataset)

# Extract the train and test splits
train_data = dataset['train']
test_data = dataset['test']

# Convert to pandas DataFrame for easy manipulation
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
'''
# Save the train DataFrame to a CSV file
train_df.to_csv('amazon_reviews.csv', index=False)
print("Train data saved to amazon_reviews.csv")
'''

# Combine train and test data
total_df = pd.concat([train_df, test_df], ignore_index=True)


'''
# Save the combined DataFrame to a CSV file
total_df.to_csv('amazon_reviews_total.csv', index=False)
print("Total data saved to amazon_reviews_total.csv")

# Print logs of total data
print("Total Data Information:")
print(total_df.info())

print("Total Data Description:")
print(total_df.describe())

print("First 5 Rows of Total Data:")
print(total_df.head())

print("Last 5 Rows of Total Data:")
print(total_df.tail())
'''

# Print logs of total data

print("First 10 Reviews:")
for index, row in total_df.head(10).iterrows():
    print(f"A customer has given a review \"{row['title']}\" and review as \"{row['label']}\".")
