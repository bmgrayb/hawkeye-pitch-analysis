import logging
import pandas as pd
import pitch_path.files as files
import pitch_path.utils.preprocessing as pp
import pitch_path.utils.features as feat
import os

logger = logging.getLogger(__name__)

JOINTS_FILE_PATH = os.path.join(files.__path__[0], "joint_ids.csv")
JOINTS_DF = pd.read_csv(JOINTS_FILE_PATH)
ALL_HANDEDNESS = ['l', 'r']


class PitcherDataProcessor:
    """Class used to handled processing a raw pitch data file in the provided format for the evaluation.
    It will take in the raw data, filter to between start and release times, and calculate features.
    """
    def __init__(self, file_name: str, is_processed_file: bool=False) -> None:
        self.file_name = file_name
        self.metadata_cols = ['astros_pitch_id', 'sched_id', 'pitcher_id', 'bats', 'throws', 'time']
        self.joints_of_interest = ['wrist', 'elbow', 'shoulder']

        # initialize all variables we want to save in the class
        self.df = None
        self.wide_df = None
        self.pitcher_df = None
        self.pitcher_features_df = None
        self.handedness = None
        self.front_leg = None
        self.joints_to_filter_to = None
        self.joint_cols_fmt = None
        self.columns_to_filter_to = None
        self.leg_lift_col_name = None
        self.sched_id = None
        self.pitcher_id = None
        self.throws = None

        # process the file provided
        if not is_processed_file:
            self.initialize_from_raw_file(file_name)
        else:
            self.initialize_from_processed_file(file_name)


    def info(self) -> None:
        logger.info(f"File name: {self.file_name}")
        logger.info(f"Pitcher id: {self.pitcher_id}")
        logger.info(f"Sched id: {self.sched_id}")
        logger.info(f"Throws: {self.throws}")
        logger.info(f"Front Leg: {self.front_leg}")

    def initialize_from_raw_file(self, file_name) -> None:
        logger.info(f"Processing pitcher file: {file_name}")
        self.df = pd.read_feather(file_name)
        df_with_joints = pd.merge(self.df, JOINTS_DF, how='inner', on='joint_type_id')

        self.wide_df = pd.pivot_table(df_with_joints,
                values=['x', 'y', 'z'],
                columns=['hawkeye', ],
                index=['astros_pitch_id', 'sched_id', 'pitcher_id', 'bats', 'throws', 'time'])\
                .reset_index()
        self.wide_df.columns = [f"{col[1]}{'_' if col[1].strip() != '' else ''}{col[0]}" for col in self.wide_df.columns.values]

        self.handedness = self.wide_df.throws.unique()[0].lower()
        self.front_leg = [x for x in ALL_HANDEDNESS if x != ALL_HANDEDNESS][0]

         # get (x,y,z) column names from handedness
        self.joints_to_filter_to = [f"{self.handedness}Shoulder", f"{self.handedness}Elbow", f"{self.handedness}Wrist", f"{self.front_leg}Knee"]
        self.joint_cols_fmt =  [f"{a}_{b}" for a in self.joints_to_filter_to for b in ['x', 'y', 'z']]

        self.metadata_cols = ['astros_pitch_id', 'sched_id', 'pitcher_id', 'bats', 'throws', 'time']
        self.columns_to_filter_to = self.metadata_cols + self.joint_cols_fmt
    
        # get the leg lift
        self.leg_lift_col_name = [x for x in self.columns_to_filter_to if 'Knee' in x and self.front_leg in x and 'z' in x][0]

        # setting other metadata columns
        self.sched_id = self.wide_df['sched_id'].unique()[0]
        self.pitcher_id = self.wide_df['pitcher_id'].unique()[0]
        self.throws = self.wide_df['throws'].unique()[0]

        self.info()
        logger.info("Finished processing.....")
    
    def initialize_from_processed_file(self, file_name) -> None:
        logger.info(f"Initializing from processed pitcher file: {file_name}")
        self.pitcher_df = pd.read_feather(file_name)

        self.handedness = self.pitcher_df.throws.unique()[0].lower()
        self.front_leg = [x for x in ALL_HANDEDNESS if x != ALL_HANDEDNESS][0]

         # get (x,y,z) column names from handedness
        self.joints_to_filter_to = [f"{self.handedness}Shoulder", f"{self.handedness}Elbow", f"{self.handedness}Wrist", f"{self.front_leg}Knee"]
        self.joint_cols_fmt =  [f"{a}_{b}" for a in self.joints_to_filter_to for b in ['x', 'y', 'z']]

        self.metadata_cols = ['astros_pitch_id', 'sched_id', 'pitcher_id', 'bats', 'throws', 'time']
        self.columns_to_filter_to = self.metadata_cols + self.joint_cols_fmt
    
        # get the leg lift
        self.leg_lift_col_name = [x for x in self.columns_to_filter_to if 'Knee' in x and self.front_leg in x and 'z' in x][0]

        # setting other metadata columns
        self.sched_id = self.pitcher_df['sched_id'].unique()[0]
        self.pitcher_id = self.pitcher_df['pitcher_id'].unique()[0]
        self.throws = self.pitcher_df['throws'].unique()[0]

        self.info()
        logger.info("Finished processing.....")

    def get_raw_pitcher_df(self) -> pd.DataFrame:
        return self.df

    def get_pitcher_df(self) -> pd.DataFrame:
        if self.pitcher_df is not None:
            return self.pitcher_df
        else:
            logger.info("Creating pitcher file df....")
            df = self.wide_df[self.columns_to_filter_to]
            pitch_df_w_ll = pp.set_leg_lift_time(df, self.leg_lift_col_name)
            pitch_df_throw = pp.set_release_point(pitch_df_w_ll)
            pitch_df_filtered = pp.filter_df_to_start_release(pitch_df_throw)
            self.pitcher_df = pp.rename_handedness_cols(pitch_df_filtered)
            return self.pitcher_df
    
    def get_pitcher_features_df(self) -> pd.DataFrame:
        if self.pitcher_features_df is not None:
            return self.pitcher_features_df
        logger.info("Creating pitcher features df...")
        if self.pitcher_df is None:
            logger.info("pitcher df is null, getting pitcher df first.")
            _ = self.get_pitcher_df()

        joint_cols = [col for col in self.pitcher_df.columns for joint in self.joints_of_interest if joint in col] 
        self.pitcher_features_df = feat.generate_features_from_pitch_df(df=self.pitcher_df, pitcher_id=self.pitcher_id, sched_id=self.sched_id, joint_cols=joint_cols, joints=self.joints_of_interest)
        return self.pitcher_features_df

                
    def save_pitcher_df(self, root_dir: str, overwrite: bool=False) -> None:
        logger.info("Saving pitcher df")
        if self.pitcher_df is None:
            logger.info("pitcher df is null, getting pitcher df first.")
            _ = self.get_pitcher_df()
        
        output_dir = f"{root_dir}/processed"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        output_file_path = f"{output_dir}/sched_id{self.sched_id}_pitcher{self.pitcher_id}"

        if not overwrite and os.path.exists(output_file_path):
            raise Exception(f"File path {output_file_path} already exists.")

        logger.info(f"Writing pitcher df to {output_file_path}")
        self.pitcher_df.to_feather(path=output_file_path)


    def save_pitcher_features_df(self, root_dir: str, overwrite: bool=False) -> None:
        logger.info("Saving pitcher features df")
        if self.pitcher_features_df is None:
            logger.info("pitcher features df is null, getting pitcher features df first.")
            _ = self.get_pitcher_features_df()
        
        output_dir = f"{root_dir}/pitcher_features"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        output_file_path = f"{output_dir}/sched_id{self.sched_id}_pitcher{self.pitcher_id}"
        if not overwrite and os.path.exists(output_file_path):
            raise Exception(f"File path {output_file_path} already exists.")

        logger.info(f"Writing pitcher features df to {output_file_path}")
        self.pitcher_features_df.to_feather(path=output_file_path)
        



    