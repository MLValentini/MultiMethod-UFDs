import pandas as pd

def merge_dataframes_by_highest_probability(df1, df2):
    # Concatenate the two dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Sort the combined dataframe by 'sourceid' and 'probability' in descending order
    combined_df_sorted = combined_df.sort_values(['sourceid', 'probability'], ascending=[True, False])
    
    # Keep the first occurrence of each 'sourceid' (which will be the one with the highest probability)
    result_df = combined_df_sorted.drop_duplicates(subset='sourceid', keep='first')
    
    # Reset the index of the resulting dataframe
    result_df = result_df.reset_index(drop=True)
    
    return result_df

# Example usage:
df1 = pd.DataFrame({'sourceid': [1, 2, 3], 'bestclassification': ['A', 'B', 'C'], 'probability': [0.8, 0.6, 0.7]})
df2 = pd.DataFrame({'sourceid': [1, 2, 3], 'bestclassification': ['B', 'A', 'C'], 'probability': [0.9, 0.5, 0.6]})
result = merge_dataframes_by_highest_probability(df1, df2)
print(result)