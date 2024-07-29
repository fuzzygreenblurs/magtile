file_path = 'regression_tracked_actions/ur/center_to_ur/segment_1.csv'  # Use the path to the uploaded file
trimmed_file_path_generated = 'regression_tracked_actions/ur/center_to_ur/trimmed/segment_1.csv'

import pandas as pd

def find_repeating_position(df, min_repeats=30):
    for i in range(len(df) - min_repeats + 1):
        repeating = True
        for j in range(min_repeats):
            if (df.iloc[i]['h_pos'], df.iloc[i]['v_pos']) != (df.iloc[i + j]['h_pos'], df.iloc[i + j]['v_pos']):
                repeating = False
                break
        if repeating:
            return (df.iloc[i]['h_pos'], df.iloc[i]['v_pos'])
    return None

def trim_dataset(file_path, min_repeats=30):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Infer the start position
    start_position = find_repeating_position(df, min_repeats)
    if not start_position:
        raise ValueError("No starting position found with the required repetitions.")

    # Trim the dataset to remove the starting repeating positions
    start_pos_found = False
    for i in range(len(df) - 2):
        if (df.iloc[i]['h_pos'], df.iloc[i]['v_pos']) == start_position and \
           (df.iloc[i+1]['h_pos'], df.iloc[i+1]['v_pos']) == start_position and \
           (df.iloc[i+2]['h_pos'], df.iloc[i+2]['v_pos']) == start_position:
            start_index = i + 2
            start_pos_found = True

    if not start_pos_found:
        raise ValueError("No starting position found with the required repetitions at the end.")

    # Trim the dataset
    trimmed_df = df[start_index - 2:].reset_index(drop=True)

    # Infer the end position from the trimmed dataset
    end_position = find_repeating_position(trimmed_df, min_repeats)
    if not end_position:
        raise ValueError("No ending position found with the required repetitions.")

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

trimmed_df_generated = trim_dataset(file_path, min_repeats=30)
trimmed_df_generated.to_csv(trimmed_file_path_generated, index=False)

print(f'Trimmed dataset saved to {trimmed_file_path_generated}')

############ WORKING ###########

# def trim_dataset(file_path, start_position, end_position):
#     # Load the dataset
#     df = pd.read_csv(file_path)

#     # Convert positions to tuples for easier comparison
#     start_position = tuple(start_position)
#     end_position = tuple(end_position)

#     # Find the final occurrence of the starting position appearing three times in a row
#     start_pos_found = False
#     for i in range(len(df) - 2):
#         if (df.iloc[i]['h_pos'], df.iloc[i]['v_pos']) == start_position and \
#            (df.iloc[i+1]['h_pos'], df.iloc[i+1]['v_pos']) == start_position and \
#            (df.iloc[i+2]['h_pos'], df.iloc[i+2]['v_pos']) == start_position:
#             start_index = i + 2
#             start_pos_found = True

#     # Ensure we include the last three rows of the starting position
#     if start_pos_found:
#         trimmed_df = df[start_index - 2:].reset_index(drop=True)
#     else:
#         return pd.DataFrame(columns=df.columns)

#     # Find the first occurrence of the ending position appearing three times in a row
#     for j in range(len(trimmed_df) - 2):
#         if (trimmed_df.iloc[j]['h_pos'], trimmed_df.iloc[j]['v_pos']) == end_position and \
#            (trimmed_df.iloc[j+1]['h_pos'], trimmed_df.iloc[j+1]['v_pos']) == end_position and \
#            (trimmed_df.iloc[j+2]['h_pos'], trimmed_df.iloc[j+2]['v_pos']) == end_position:
#             end_index = j
#             break

#     # Include rows up to the end index and include the first three rows of the ending position
#     final_trimmed_df = trimmed_df[:end_index + 3]

#     return final_trimmed_df

# trimmed_df_generated = trim_dataset(file_path, start_position, end_position)
# trimmed_df_generated.to_csv(trimmed_file_path_generated, index=False)
# print(f'Trimmed dataset saved to {trimmed_file_path_generated}')
