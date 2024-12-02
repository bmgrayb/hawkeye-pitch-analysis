from functools import reduce
import logging
import numpy as np
import pandas as pd
import pitch_path.utils.preprocessing as pp
from scipy.spatial import distance


logger = logging.getLogger(__name__)

def get_col_values_at_time(df: pd.DataFrame, time_col: str, joint_loc_cols: list) -> pd.DataFrame:
    """Helper function to return a list of columns at different time intervals. E.g. at start, release, midway point, etc.
    This works by grabbing the rows where the time column is 1.

    Args:
        df (pd.DataFrame): pitch dataframe
        time_col (str): time column to filter to (start, release, etc)
        joint_loc_cols (list): which joint columns to return

    Returns:
        pd.DataFrame: pitch dataframe filtered to the columns at the desired time
    """
    loc_df = df.loc[df[time_col] == 1, joint_loc_cols].reset_index().drop(columns=['index'])
    loc_df.columns = [f"{x}_{time_col}" for x in loc_df.columns]
    return loc_df

def get_distance_to_prev(df: pd.DataFrame, joint_cols: list, prev_joint_cols: list, distance_col_name: str="distance_to_prev") -> pd.DataFrame:
    """Helper function to calculate the euclidean distance to the previous point. This can be used to cacluate total distance, 
    average velocity, or any other feature that needs distance.

    Args:
        df (pd.DataFrame): pitch dataframe
        joint_cols (list): list of columns at point A all of the same joint
        prev_joint_cols (list): list of columnas at point B all of the same joint
        distance_col_name (str, optional): new column name to store distance. Defaults to "distance_to_prev".

    Returns:
        pd.DataFrame: pitch dataframe with added distance column
    """
    df[distance_col_name] = df.apply(lambda x: 0 if np.isnan(x[prev_joint_cols[0]])
                                  else distance.cdist([x[joint_cols].to_list()], [x[prev_joint_cols].to_list()], 'euclidean')[0][0], axis=1)
    return df

def get_avg_velocity_between_points(df: pd.DataFrame, distance_col_name: str="distance_to_prev") -> pd.DataFrame:
    """Helper function to calculate the average velocity to the previous point. Requires both time, prev_time, 
    and distance column to be in the provided dataframe

    Args:
        df (pd.DataFrame): pitch dataframe
        distance_col_name (str, optional): distance column used to calculate average velocity to previous point. Defaults to "distance_to_prev".

    Raises:
        Exception: missing necessary columns

    Returns:
        pd.DataFrame: pitch dataframe with added velocity column
    """
    if distance_col_name not in df.columns or 'time' not in df.columns or 'prev_time' not in df.columns:
        raise Exception("Missing necessary columns to calculate velocity...")
    df['avg_velocity'] = df.apply(lambda x: x[distance_col_name]/(x['time'] - x['prev_time']), axis=1)
    return df

def get_distance_traveled_for_pitch(df: pd.DataFrame, distance_col_name: str, output_col_name: str) -> pd.DataFrame:
    """helper function to get the total distance traveled in a give column

    Args:
        df (pd.DataFrame): pitch dataframe
        distance_col_name (str): column to sum up total distance
        output_col_name (str): new column name

    Returns:
        pd.DataFrame: pitch dataframe with total distance column added
    """
    return pd.DataFrame(data={output_col_name: [df[distance_col_name].sum()]})


def generate_feature_row_from_pitch_df(df: pd.DataFrame, pitcher_id: int, sched_id: int, all_joint_cols: list, all_prev_joint_cols: list, joints: list, distance_col_name: str="distance_to_prev") -> pd.DataFrame:
    """Helper function to generate all features for a given pitch. The pitch dataframe is required to have the same shared astros_pitch_id (e.g. single pitch in a game)
    for the features to be calculated properly. This will return a singular row with all features for given pitch.

    Args:
        df (pd.DataFrame): pitch dataframe for a singular pitch
        pitcher_id (int): pitchid of pitch
        sched_id (int): schedule id of pitch
        all_joint_cols (list): list of join columns to added features to
        all_prev_joint_cols (list): list of joint columns to add features required from previous row
        joints (list): list of joints to calculate features for
        distance_col_name (str, optional): base distance column name. Defaults to "distance_to_prev".

    Returns:
        pd.DataFrame: pitch feature dataframe that contains a singular feature row for each pitch
    """

    joint_features_list = []

    for j in joints:
        joint_cols = [x for x in all_joint_cols if j in x]
        prev_joint_cols = [x for x in all_prev_joint_cols if j in x]

        # add addtional features
        df_distance_prev = get_distance_to_prev(df, joint_cols, prev_joint_cols, f"{distance_col_name}_{j}")
        df_velo = get_avg_velocity_between_points(df_distance_prev,  f"{distance_col_name}_{j}")
        
        # features used to generate singular feature row
        df_dist = get_distance_traveled_for_pitch(df_velo,  f"{distance_col_name}_{j}", f"distance_traveled_{j}")
        joint_start = get_col_values_at_time(df, 'start', joint_cols)
        joint_25 = get_col_values_at_time(df, 'time_25', joint_cols)
        joint_5 = get_col_values_at_time(df, 'time_5', joint_cols)
        joint_75 = get_col_values_at_time(df, 'time_75', joint_cols)
        joint_release = get_col_values_at_time(df, 'release', joint_cols)
        
        joint_features_list.append(df_dist.merge(joint_start, how='cross')\
            .merge(joint_25, how='cross')\
            .merge(joint_5, how='cross')\
            .merge(joint_75, how='cross')\
            .merge(joint_release, how='cross'))
        
    # merge all features to a single row and add metadata columns
    pitch_feature = reduce(lambda x, y: x.merge(y, how='cross'), joint_features_list)
    pitch_feature['pitcher_id'] = pitcher_id
    pitch_feature['sched_id'] = sched_id
    pitch_feature['astros_pitch_id'] = df.astros_pitch_id.unique()[0]

    return pitch_feature

def generate_features_from_pitch_df(df: pd.DataFrame, pitcher_id: int, sched_id: int, joint_cols: list, joints: list, distance_col_name: str="distance_to_prev") -> pd.DataFrame:
    """Function to generate all features from a given game pitch dataframe.

    Args:
        df (pd.DataFrame): pitch dataframe that has been filtered to start and relese 
        pitcher_id (int): pitchid of pitch
        sched_id (int): schedule id of pitch
        joint_cols (list): list of joint columns to added features to
        joints (list): list of joints to calculate features for
        distance_col_name (str, optional): base distance column name. Defaults to "distance_to_prev".

    Returns:
        pd.DataFrame: pitch feature dataframe that contains a feature row for all pitches
    """
    logger.info(f"Getting features for pitcher_id: {pitcher_id} and sched_id: {sched_id}")
    all_pitches_features_list = []

    prev_joint_cols = [f"prev_{x}" for x in joint_cols]
    cols_to_shift = joint_cols + ['time']

    for pitch in df['astros_pitch_id'].unique():
        pitch_df = df[df['astros_pitch_id'] == pitch]
        pitch_df = pp.add_shifted_columns(pitch_df, cols_to_shift)

        feature_row = generate_feature_row_from_pitch_df(pitch_df, pitcher_id, sched_id, joint_cols, prev_joint_cols, joints, distance_col_name)
        all_pitches_features_list.append(feature_row)

    return pd.concat(all_pitches_features_list)