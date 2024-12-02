import pandas as pd
import logging
logger = logging.getLogger(__name__)

def set_release_point(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to get set the release point feature flag of a dataframe for each pitch in a dataframe.
    The expectation is that the dataframe has a unique key on `astros_pitch_id`.

    Args:
        df (pd.DataFrame): Dataframe contain pitch data

    Returns:
        pd.DataFrame: Dataframe with release column at index containing the release point
    """
    logger.info("Setting release point....")
    min_index = df.groupby('astros_pitch_id', as_index=False).agg({'time': lambda x: x.abs().idxmin()})['time'].values
    df['release'] = 0
    df.loc[min_index, 'release'] = 1
    return df

def get_leg_lift_time(df: pd.DataFrame, col:str = 'z', window:int = 30) -> pd.DataFrame:
    """Helper function to calculate the rolling window where the given value has continually increased for window size rows. This is used to find the
    index where we can say a leg lift has started so it can be flagged.

    Args:
        df (pd.DataFrame): pitch dataframe
        col (str, optional): column to calulate rolling window against. Defaults to 'z'.
        window (int, optional): size of window rquired to be monotonically increasing. Defaults to 30.

    Returns:
        pd.DataFrame: time interval where we set as the leg lift start
    """
    df_sorted = df.sort_values(by=['time']).reset_index()
    df_sorted['increasing'] = df_sorted[col].diff().gt(0).rolling(window).sum().eq(window)
    leg_lift_time = df_sorted[df_sorted['increasing']].time.min()
    return leg_lift_time

def set_leg_lift_time(df: pd.DataFrame, leg_lift_col: str) -> pd.DataFrame:
    """Function to set the time in the pitch dataframe where we say a leg lift has begun. This will have a new column added called `start` that will
    be set to 1 at the index where the leg lift started, and 0 everywhere else.

    Args:
        df (pd.DataFrame): pitch dataframe
        leg_lift_col (str): column to monitor for monotonic increase, e.g. z coordinate

    Returns:
        pd.DataFrame: _description_
    """
    logger.info("Setting leg lift time....")
    astros_pitch_id = df.astros_pitch_id.unique()
    leg_lift_time = []
    for pitch_id in astros_pitch_id:
        pitch_df = df[(df.astros_pitch_id == pitch_id)]
        leg_lift_time.append(get_leg_lift_time(pitch_df, leg_lift_col))
    
    ll = pd.DataFrame({'astros_pitch_id': astros_pitch_id, 'time': leg_lift_time})
    ll['start'] = 1

    pitches_w_start = pd.merge(df, ll, how='left', on=['astros_pitch_id', 'time']).fillna(0)

    return pitches_w_start

def filter_df_to_start_release(df: pd.DataFrame) -> pd.DataFrame:
    """function to filter dataframe to be just the leg lift to the release. This is so we can isolate the throwing
    motion to calculate features on only the valid part.

    Args:
        df (pd.DataFrame): pitch data frame

    Returns:
        pd.DataFrame: pitch dataframe that has been filtered to start and release.
    """
    logger.info("Filtering df to start and release times.")
    filtered_dfs = []
    for pitch in df.astros_pitch_id.unique():
        pitch_no_df = df[(df['astros_pitch_id'] == pitch)]
        pitch_start = pitch_no_df[pitch_no_df['start'] == 1].index[0]
        pitch_release = pitch_no_df[pitch_no_df['release'] == 1].index[0]
        pitch_df_fil = pitch_no_df.loc[pitch_start:pitch_release].reset_index().drop(columns=['index'])
        pitch_df_with_percentiles = set_time_percentiles(pitch_df_fil)
        filtered_dfs.append(pitch_df_with_percentiles)
    full_filtered_df = pd.concat(filtered_dfs)
    full_filtered_df['sched_id'] = full_filtered_df['sched_id'].astype(int)
    full_filtered_df['astros_pitch_id'] = full_filtered_df['astros_pitch_id'].astype(int)
    full_filtered_df['pitcher_id'] = full_filtered_df['pitcher_id'].astype(int)

    return full_filtered_df

def set_time_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """ Create new columns to designate rows for the 25th, 50th, and 75th time percentiles. This is so we can grab
    different join (x,y,z) locations at various points along the path.

    Args:
        df (pd.DataFrame): pitch dataframe

    Returns:
        pd.DataFrame: pitch dataframe with new columns for the 25, 50, and 75 time percentiles
    """
    # get the index for the 25, 50, and 75 percentile by time (DF should be ordered by time already)
    percentile_25_id = int(0.25 * df.shape[0])
    percentile_5_id = int(0.5 * df.shape[0])
    percentile_75_id = int(0.75 * df.shape[0])

    # set column values to 1 for corresponding IDs
    df['time_25'] = 0
    df.loc[percentile_25_id, 'time_25'] = 1
    df['time_5'] = 0
    df.loc[percentile_5_id, 'time_5'] = 1
    df['time_75'] = 0
    df.loc[percentile_75_id, 'time_75'] = 1

    return df

def rename_handedness_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to rename the pitch dataframe columns to remove any handedness so we can have
    a standard format for features built for throwing path.

    Args:
        df (pd.DataFrame): pitch dataframe

    Returns:
        pd.DataFrame: pitch dataframe with columns renamed
    """
    renamed_cols = [x.lower() if ((not x.startswith("l") and not x.startswith('r')) or x is 'release') else x[1:].lower() for x in df.columns]
    df.columns = renamed_cols
    return df

def add_shifted_columns(df: pd.DataFrame, cols_to_shift: list) -> pd.DataFrame:
    """Helper function to add shifted columns to the current row. This allows us to calculate features that 
    need the (x,y,z) coordinates of the previous row

    Args:
        df (pd.DataFrame): pitch dataframe
        cols_to_shift (list): list of columns to shift 1

    Returns:
        pd.DataFrame: pitch dataframe with shifted columns added
    """
    for col in cols_to_shift:
        df[f"prev_{col}"] = df[col].shift(1)

    return df