import numpy as np
import pandas as pd
import random
from PIL import Image

class Animal_grouper:
    WEIGHT_COLUMN = "Weight"
    SEX_COLUMN = "Sex"
    MAX_RETRIES = 1000
    
    def __init__(self, df: pd.core.frame.DataFrame, group_amount: int, threshold: float, is_based_on_gender: bool):
        """
        Initialize the AnimalGrouper object.

        Args:
            df (pd.core.frame.DataFrame): The DataFrame containing animal data.
            group_amount (int): The number of groups to create.
            threshold (float): The threshold for standard deviation.
            is_based_on_gender (bool): Whether to consider gender when grouping.
        """
        # Error handle: input validation
        if self.WEIGHT_COLUMN not in df.columns or self.SEX_COLUMN not in df.columns:
            raise ValueError("Uploaded data must contain 'Weight' and 'Sex' columns.")
           
        self.df = df
        self.group_amount = group_amount
        self.threshold = threshold
        self.is_based_on_gender = is_based_on_gender
        self.weight_data = np.array(self.df[self.WEIGHT_COLUMN]) # return a Series
        
        if self.is_based_on_gender:
            self.result = self.group_based_on_deviation_and_gender()
            
        else:
            self.result = self.group_based_on_deviation(self.weight_data)
            
        self.output = self.convert_to_df(self.result)
        
        
    def distribute_data_evenly(self, datas: np.ndarray):
        """
        Distributes data evenly into groups.
    
        Args:
            datas (np.ndarray): The data to be distributed.
    
        Returns:
            list: A list of lists representing the grouped data.
        """
        # create group_amount lists in a list
        # _ means to other developers that the variable is unused
        groups = [[] for _ in range(self.group_amount)]
        
        # put each data into the containers randomly    
        for i, data in enumerate(datas):
            group_index = i % self.group_amount
            groups[group_index].append(data)
            
        return groups
            
            
    def group_randomly(self, datas: np.ndarray) -> list:
        """
        Group the ndarray data randomly into group_number.
        
        Args:
            data (np.ndarray): The data to be grouped.

        Returns:
            list: A list of lists representing the grouped data.
        """
        # get the max_animal_per_group
        max_animal_per_group = len(datas) // self.group_amount
        if len(datas) % self.group_amount != 0:
            max_animal_per_group += 1
        
        # Shuffle the datas
        random.shuffle(datas)
        
        return self.distribute_data_evenly(datas)
    
    
    def check_std_deviation_within_threshold(self, grouped_data: list) -> bool:
        """
        Check if the standard deviation of each group is within the allowed threshold.

        Args:
            grouped_data (list): A list of lists representing grouped data.

        Returns:
            bool: True if all group standard deviations are within the threshold, False otherwise.
        """
        # Calculate the standard deviation of each group
        std_of_each_group = np.array([np.std(group) for group in grouped_data])

        # Check if any group's standard deviation exceeds the threshold
        return np.any(std_of_each_group > np.std(grouped_data) * self.threshold)
    
    
    def group_based_on_deviation(self, data: np.ndarray) -> list:
        """
        Group the data based on weight while ensuring standard deviation falls within the threshold.

        Args:
            data (np.ndarray): The data to be grouped.

        Returns:
            list: A list of lists representing the grouped data. If retries exceed the maximum limit,
            a custom exception is raised.
        """
        grouped_list = self.group_randomly(data) # return a list
        counter = 0
        while self.check_std_deviation_within_threshold(grouped_list):
            grouped_list = self.group_randomly(data)
            counter += 1
            if counter > self.MAX_RETRIES:
                raise OverRetryException(
                    f"Tried {self.MAX_RETRIES} times. The deviation is too big.\n"
                )

        return grouped_list
    
    
    def combine_lists_within_threshold(self, list1: list, list2: list) -> list:
        """
        Combines two lists while ensuring the standard deviation falls within the threshold.
    
        Args:
            list1 (list): The first list of data.
            list2 (list): The second list of data.
    
        Returns:
            list: A list representing the combined data.
        """
        # combine two grouped data
        combined_list = []
        group_label = list(range(self.group_amount))
        
        counter = 0
        is_within_threshold = True
        
        while is_within_threshold:
            combined_list = []
            group_label = list(range(self.group_amount))
            counter += 1

            for i in range(self.group_amount):
                number = random.choice(group_label)
                combined_list.append(
                    list1[i] + list2[number]
                )
                group_label.pop(group_label.index(number))

            if counter > self.MAX_RETRIES:
                raise OverRetryException(
                    f"Tried {self.MAX_RETRIES} times. The deviation is too big.\n"
                )
            is_within_threshold = self.check_std_deviation_within_threshold(combined_list)
        
        return combined_list
    
    def group_based_on_deviation_and_gender(self) -> list:
        """
        Groups data based on weight and gender, ensuring standard deviation falls within the threshold.
    
        Returns:
            np.ndarray: A NumPy array representing the grouped data.
        """
        female_df = self.df[self.df["Sex"] == "F"]
        female_weight_ndarray = np.array(female_df["Weight"])
        male_df = self.df[self.df["Sex"] == "M"]
        male_weight_ndarray = np.array(male_df["Weight"])

        grouped_female = self.group_based_on_deviation(female_weight_ndarray)
        grouped_male = self.group_based_on_deviation(male_weight_ndarray)
        
        grouped_both_genders = self.combine_lists_within_threshold(grouped_female, grouped_male)
            
        self.output_female = self.convert_to_df(grouped_female)
        self.output_male = self.convert_to_df(grouped_male)
        
        return grouped_both_genders
    
    
    def convert_to_df(self, grouped_list: list) -> pd.DataFrame:
        """
        Converts a grouped NumPy array into a Pandas DataFrame.
    
        Args:
            grouped_ndarray (np.ndarray): The grouped data as a NumPy array.
    
        Returns:
            pd.DataFrame: A Pandas DataFrame representing the grouped data.
        """
        
        grouped_df = pd.DataFrame()
        
        # Error handle
        num_groups = len(grouped_list)
        if num_groups < self.group_amount:
            raise ValueError(f"Expected {self.group_amount} groups, but got {num_groups}.")
    
        for i in range(num_groups):
            grouped_df[f"Group {i+1}"] = grouped_list[i]
    
        return grouped_df
    

class OverRetryException(Exception):
    """
    Raised when the retry limit is exceeded.
    
    Attributes:
        message (str): A descriptive error message.
    """

    def __init__(self, message=""):
        """
        Initialize the exception with an optional error message.

        Args:
            message (str, optional): A descriptive error message. Defaults to an empty string.
        """
        self.message = message
        super().__init__(self.message)
        

if __name__ == "__main__":
    df = pd.read_excel("./data/data_sample.xlsx")
    try:
        ag = Animal_grouper(df, 5, 1.3, True)
        df = ag.output_female
        print(df)
    
    except ValueError:
        print("Please make sure you upload the correct format file.")
        
    except OverRetryException:
        print("The deviation of the data is too big. Maybe adjust the threshold value.")
        
    except Exception:
        print("Oh no... Something bad happened. Please contact me.")
        
    
    
    
    
    
    
    
    
    
    