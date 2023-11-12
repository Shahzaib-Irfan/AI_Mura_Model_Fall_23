import numpy as np
import pandas as pd

class Data(object):
    
    """
        Represents a pandas instance of the dataframes from the dataset. This has 
        four properties:
            • train_df : returns the instance of train paths.
            • valid_df : returns the instance of validation paths.
            • train_labels_data : returns the instance of labeled training data.
            • valid_labels_data : returns the instance of labeled validation data.
    """
    