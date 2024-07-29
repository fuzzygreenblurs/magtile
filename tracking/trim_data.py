file_path = 'regression_tracked_actions/ur/center_to_ur/segment_10.csv'  # Use the path to the uploaded file
trimmed_file_path_generated = 'regression_tracked_actions/ur/center_to_ur/trimmed/segment_10.csv'

# Specify the starting and ending positions
start_position = [2.38,-0.74]  # Replace with your starting position
end_position   = [1.57,-1.59]     # Replace with your ending position

import pandas as pd

def trim_dataset(file_path, start_position, end_position):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Convert positions to tuples for easier comparison
    start_position = tuple(start_position)
    end_position = tuple(end_position)

    # Find the final occurrence of the starting position appearing three times in a row
    start_pos_found = False
    for i in range(len(df) - 2):
        if (df.iloc[i]['h_pos'], df.iloc[i]['v_pos']) == start_position and \
           (df.iloc[i+1]['h_pos'], df.iloc[i+1]['v_pos']) == start_position and \
           (df.iloc[i+2]['h_pos'], df.iloc[i+2]['v_pos']) == start_position:
            start_index = i + 2
            start_pos_found = True

    # Ensure we include the last three rows of the starting position
    if start_pos_found:
        trimmed_df = df[start_index - 2:].reset_index(drop=True)
    else:
        return pd.DataFrame(columns=df.columns)

    # Find the first occurrence of the ending position appearing three times in a row
    for j in range(len(trimmed_df) - 2):
        if (trimmed_df.iloc[j]['h_pos'], trimmed_df.iloc[j]['v_pos']) == end_position and \
           (trimmed_df.iloc[j+1]['h_pos'], trimmed_df.iloc[j+1]['v_pos']) == end_position and \
           (trimmed_df.iloc[j+2]['h_pos'], trimmed_df.iloc[j+2]['v_pos']) == end_position:
            end_index = j
            break

    # Include rows up to the end index and include the first three rows of the ending position
    final_trimmed_df = trimmed_df[:end_index + 3]

    return final_trimmed_df

trimmed_df_generated = trim_dataset(file_path, start_position, end_position)
trimmed_df_generated.to_csv(trimmed_file_path_generated, index=False)
print(f'Trimmed dataset saved to {trimmed_file_path_generated}')
