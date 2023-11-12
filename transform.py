from load_data import Data
import warnings
warnings.filterwarnings('ignore')

class SingularDataframe(object):
    """
        Represents a joint instance of the studies, body parts, labels, image paths
        in a singular dataframe.
    """

    @property
    def labeled_train(self):
        """
            Return a joint Singular Dataframe of both the valid and train dataframes.
            
            Returns:
                valid_df : The dataframe of validation images.
                train_df : The dataframe of training images.
        """
        
        df = Data()
        
        train_df, valid_df = df.train_df, df.valid_df
        train_labels_data = df.train_labels_data
        valid_labels_data = df.valid_labels_data
        
        # print(train_df.head())

        
        train_df['Label'] = train_df.apply(lambda f: 1 if 'positive' in f.FilePath else 0, axis=1)
        train_df['BodyPart'] = train_df.apply(lambda f: (f.FilePath.split('/')[2][3:]).lower(), axis=1)
        train_df['StudyType'] = train_df.apply(lambda f: f.FilePath.split('/')[4][:6], axis=1)
        #print(train_df.head(25))
        
        
        valid_df['Label'] = valid_df.apply(lambda f: 1 if 'positive' in f.FilePath else 0, axis=1)
        valid_df['BodyPart'] = valid_df.apply(lambda f: (f.FilePath.split('/')[2][3:]).lower(), axis=1)
        valid_df['StudyType'] = valid_df.apply(lambda f: f.FilePath.split('/')[4][:6], axis=1)
        
        
        train_df.set_index(["FilePath", "BodyPart"]).count(level="BodyPart")

        train_df.set_index(["FilePath", "Label"]).count(level="Label")
        
        return valid_df, train_df
    
            
    @property
    def df_paths_adjusted(self):
        """
            Return the dataframes, but with the path ../ adjusted in it.
        """
        valid_df,train_df = self.labeled_train
        train_df['FilePath'] = train_df.apply(lambda f: '../'+f.FilePath, axis=1)
        valid_df['FilePath'] = valid_df.apply(lambda f: '../'+f.FilePath, axis=1)
        return valid_df, train_df
        
if __name__ == '__main__':
    df = SingularDataframe()
    v,t = df.labeled_train
    print(v.head(1))