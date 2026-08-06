"""Convert legacy user selections into adjusted YOLO label files."""

import pandas as pd
import os
from .clicker_utils import append_yolov5_label, get_relative_label_name

def append_modified_labels(df):
    """Write adjusted legacy selection rows from a dataframe to label files."""
    # Group by 'ImageName' and 'SeriesID'
    grouped = df.groupby(['PathName', 'SeriesID'])
    for (label_path, series_id), group in grouped:
        print(f"Processing Group: ImageName={label_path}, SeriesID={series_id}")
        label_path = label_path.replace("labels", "inspectlabels")

        for index, row in group.iterrows():
#             print(f"Processing Group: ImageName={label_path}, SeriesID={series_id}")
            x_center= float(row['XCenter'])
            y_center= float(row['YCenter'])
            width= float(row['Width'])
            height= float(row['Height'])
            img_width = img_height = 1
            class_id= int(row['ClassID'])

            if os.path.exists(label_path):
                append_yolov5_label(label_path, x_center, y_center, width, height, img_width, img_height, class_id)
            
            label_path = get_relative_label_name(label_path, 1)
            
def modify_class_ids(df, selected_indices_df, target_class_id):
    """Replace class IDs for rows selected in a user-selection dataframe."""
    # Normalize the path names in both DataFrames
    df['PathName'] = df['PathName'].apply(lambda x: x.replace('\\', '/'))
    selected_indices_df['PathName'] = selected_indices_df['PathName'].apply(lambda x: x.replace('\\', '/'))

    # Ensure SeriesID is consistent in type (e.g., both as strings)
    df['SeriesID'] = df['SeriesID'].astype(str)
    selected_indices_df['SeriesID'] = selected_indices_df['SeriesID'].astype(str)

    modified_data = []



    for _, row in selected_indices_df.iterrows():
        path_name = row['PathName']
        series_id = row['SeriesID']
        selected_index = row['SelectedIndex']
        if selected_index != -1:
            

#             print(f"Processing PathName: {path_name}, SeriesID: {series_id}, SelectedIndex: {selected_index}")

            group = df[(df['PathName'] == path_name) & (df['SeriesID'] == str(series_id))]
#             print("Filtered Group:\n", group)  # Debugging: print the filtered group

            if not group.empty:
#                 print("Filtered Group:\n", group)  # Debugging: print the filtered group

                # Calculate new ClassIDs
    #             new_class_ids = list(range(len(group) - 1, selected_index - 1, -1)) + \
    #                             [target_class_id] + \
    #                             list(range(1, len(group) - selected_index))


    #             selected = 5
    #             0 1 2 3 4 5 6 7 8 becomes
    #             7 6 5 4 3 2 1 0

    #             selected = 6
    #             0 1 2 3 4 5 6 7 8 becomes
    #             8 7 6 5 4 3 2 1 0

    #             print(selected_index, target_class_id)

                new_class_ids = list(reversed(list(range(selected_index + target_class_id+1))))
    #             print(new_class_ids)
    #             print(len(group['ClassID']))
    #             # Truncate the group if new_class_ids is shorter than the length of the group
                if len(new_class_ids) < len(group):
                    group = group.iloc[:len(new_class_ids)]
                elif len(new_class_ids) > len(group):
                    new_class_ids = new_class_ids[:len(group)]

                # Update the group with new class IDs
                group = group.copy()
                group['ClassID'] = new_class_ids


                modified_data.append(group)

        # Concatenate all modified groups into a new DataFrame
    modified_df = pd.concat(modified_data, ignore_index=True)

    return modified_df


