import os
from pathlib import Path
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt


class DyslexiaVizualization:
    """
    A class to create 3D representation data types.

    Attributes:
        shape (tuple): The shape of the visualization area.
        dataset_name (str): The name of the dataset file.
        sheet_names (list): List of sheet indices or names to read from the Excel file.
        path (Path): The path to the dataset file.
        file_format (str): The format of the dataset file ('xlsx', 'csv').
    """
    def __init__(self, shape, dataset_name, sheet_name: list = [0,1,2],  path: Path= Path("../Datasets"), file_format: str = "xlsx") -> None:
        self.shape = shape  
        self.x_coord_norm = self.shape[1]/1500
        self.y_coord_norm = self.shape[0]/600

        self.sheet_names = sheet_name
        self.fixation = os.path.join(path, dataset_name)
        self.file_format = file_format

        self.fix_data = None
        self.seq_length = 335  

        self.cols = [(102,102,255), (102,178,255), (102, 255, 178), (178, 255, 102), 
                (255,255,102), (255,102,102), (255, 102, 178), (255,102, 255), (178,102,255)]

        self.labels = []
        self.features = []

    def __Fixation_dataset(self):
        """
        Loads the fixation dataset based on the specified file format.
        """
        if self.file_format == "xlsx":
            fix_xl = pd.read_excel(self.fixation, sheet_name=self.sheet_names)
            self.fix_data = pd.concat([fix_xl[i] for i in fix_xl.keys()], ignore_index=True, sort=False)
        elif self.file_format == "csv":
            self.fix_data = pd.read_csv(self.fixation)

    def __id_getter(self):
        return self.fix_data['SubjectID'].unique()
        
    def size_change_video_creation(self, norm_size: int = 3, shift: int = 6, description: bool = False, padding: bool = False):
        """
        Creates a time-encoded marker-based representation.

        Parameters:
            norm_size (int): Normalization size for fixation duration.
            shift (int): Vertical shift for fixation points.
            description (bool): Whether to add descriptive text to the frames.
            padding (bool): Whether to pad sequences to a fixed length.
        """
        image = np.zeros([self.shape[0],self.shape[1],3],dtype=np.uint8)
        image.fill(255)

        self.__Fixation_dataset()

        fix_norm = self.fix_data.copy()
        x = fix_norm['FIX_DURATION'].values.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        fix_norm['FIX_DURATION'] = (x_scaled * norm_size) + 2

        ids = self.__id_getter()
        labels = []

        for id in ids:
            frames_l = []
            df_id = fix_norm[fix_norm['SubjectID'] == id].iloc[-20:]
            num = 0
            for _, row in df_id.iterrows():
                image_copy = image.copy()
                if description:
                    cv.putText(image_copy, f"Sentence ID: {row['Sentence_ID']}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                    cv.putText(image_copy, f"Word Number: {row['Word_Number']}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                center_coordinates = (int(row['FIX_X'] * self.x_coord_norm), int(row['FIX_Y']* self.y_coord_norm - shift))
                cv.circle(image_copy, center_coordinates, int(row['FIX_DURATION']), self.cols[row['Word_Number'] - 1], -1)
                num += 1
                #frame normalization
                #resized_frame = cv.resize(image_copy, (self.norm_shape[0], self.norm_shape[1]))
                normalized_frame = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
                normalized_frame = normalized_frame / 255
                frames_l.append(normalized_frame)
            if padding:
                image_copy = image.copy()
                while num < self.seq_length:
                    #resized_frame = cv.resize(image_copy, (self.norm_shape[0], self.norm_shape[1]))
                    normalized_frame = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
                    normalized_frame = normalized_frame / 255
                    frames_l.append(normalized_frame)
                    num += 1
            self.features.append(frames_l[:self.seq_length])
            labels.append(df_id['Group'].unique()[0] - 1)
            self.labels = tf.keras.utils.to_categorical(labels)

    def trajectory_creation(self, norm_size: int = 3, shift: int = 6, description: bool = False, padding: bool = False):
        """
        Creates a trajectory tracking using connecting lines representation.

        Parameters:
            norm_size (int): Normalization size for fixation duration.
            shift (int): Vertical shift for fixation points.
            description (bool): Whether to add descriptive text to the frames.
            padding (bool): Whether to pad sequences to a fixed length.
        """
        image = np.zeros([self.shape[0], self.shape[1], 3],dtype=np.uint8)
        image.fill(255)

        self.__Fixation_dataset()

        fix_norm = self.fix_data.copy()
        x = fix_norm['FIX_DURATION'].values.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        fix_norm['FIX_DURATION'] = (x_scaled * norm_size) + 2 

        ids = self.__id_getter()
        labels = []

        for id in ids:
            frames_l = []
            df_id = fix_norm[fix_norm['SubjectID'] == id].iloc[-20:]
            num = 0
            image_copy = image.copy()
            prev_point = False
            for _, row in df_id.iterrows():
                if description:
                    cv.putText(image_copy, f"Sentence ID: {row['Sentence_ID']}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                    cv.putText(image_copy, f"Word Number: {row['Word_Number']}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                center_coordinates = (int(row['FIX_X'] * self.x_coord_norm), int(row['FIX_Y']* self.y_coord_norm - shift))
                #print(center_coordinates)
                if prev_point:
                    cv.line(image_copy, prev_point, center_coordinates, self.cols[row['Word_Number'] - 1],1)
                    prev_point =  center_coordinates
                else:
                    prev_point = center_coordinates
                cv.circle(image_copy, center_coordinates, int(row['FIX_DURATION']), self.cols[row['Word_Number'] - 1], -1)
                normalized_frame = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
                normalized_frame = normalized_frame / 255
                frames_l.append(normalized_frame)
            if padding:
                image_copy = image.copy()
                while num < self.seq_length:
                    #resized_frame = cv.resize(image_copy, (self.norm_shape[0], self.norm_shape[1]))
                    normalized_frame = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
                    normalized_frame = normalized_frame / 255
                    frames_l.append(normalized_frame)
                    num += 1
            self.features.append(frames_l[:self.seq_length])
            labels.append(df_id['Group'].unique()[0] - 1)
            self.labels = tf.keras.utils.to_categorical(labels)
            
    def huddled_creation(self, norm_size: int = 4, shift: int = 25, moving: int = 3, description: bool = False, padding: bool = False):
        """
        Creates a multi-level markers representation.

        Parameters:
            norm_size (int): Normalization size for fixation duration.
            shift (int): Vertical shift for fixation points.
            moving (int): Incremental shift per word number.
            description (bool): Whether to add descriptive text to the frames.
            padding (bool): Whether to pad sequences to a fixed length.
        """
        image = np.zeros([self.shape[0], self.shape[1], 3],dtype=np.uint8)
        image.fill(255)

        self.__Fixation_dataset()

        fix_norm = self.fix_data.copy()
        x = fix_norm['FIX_DURATION'].values.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        fix_norm['FIX_DURATION'] = (x_scaled * norm_size) + 2 

        ids = self.__id_getter()
        labels = []

        for id in ids:
            frames_l = []
            df_id = fix_norm[fix_norm['SubjectID'] == id].iloc[-20:]
            num = 0
            image_copy = image.copy()
            for _, row in df_id.iterrows():
                if description:
                    cv.putText(image_copy, f"Sentence ID: {row['Sentence_ID']}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                    cv.putText(image_copy, f"Word Number: {row['Word_Number']}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                center_coordinates = (int(row['FIX_X'] * self.x_coord_norm), int(row['FIX_Y']* self.y_coord_norm - (shift - moving*(row['Word_Number'] - 1))))
                cv.circle(image_copy, center_coordinates, int(row['FIX_DURATION']), self.cols[row['Word_Number'] - 1], -1)
                normalized_frame = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
                normalized_frame = normalized_frame / 255
                frames_l.append(normalized_frame)
            if padding:
                image_copy = image.copy()
                while num < self.seq_length:
                    #resized_frame = cv.resize(image_copy, (self.norm_shape[0], self.norm_shape[1]))
                    normalized_frame = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
                    normalized_frame = normalized_frame / 255
                    frames_l.append(normalized_frame)
                    num += 1
            self.features.append(frames_l[:self.seq_length])
            labels.append(df_id['Group'].unique()[0] - 1)
            self.labels = tf.keras.utils.to_categorical(labels)

    def get_datas(self, type_: str = "by_size"):
        """
        Generates features and labels based on the specified type.

        Parameters:
            type_ (str): Type of data creation method to use. Options are "by_size", "traj", or "huddle".
    
        Returns:
            tuple: A tuple containing the features and labels arrays.
        """
        self.features, self.labels = [], []
        if type_ == "by_size":
            self.size_change_video_creation()
        elif type_ == "traj":
            self.trajectory_creation()
        elif type_ == "huddle":
            self.huddled_creation()

        self.features = np.asarray(self.features)
        self.labels = np.array(self.labels)
        return self.features, self.labels