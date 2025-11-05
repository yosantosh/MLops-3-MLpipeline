import pandas as pd, numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import os
import logging




#------------------------------------------------------------------------------------------------
                                     # setup logger #


# first we need to create a dir as name log
log_dir =  "/home/santosh/Desktop/MLOps/Class_3_ML_Pipeline/Log"
os.makedirs(log_dir, exist_ok=True)      #it will create dir called log if ist not already exists 


logger = logging.getLogger('Data preprocessing') # creating logger object and Data_Ingetion is name of this logger object that we will gonna use it later
logger.setLevel('DEBUG')   # all log levels will be convered in debug level of logger



# setup logger handler 
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

Log_file_path = os.path.join(log_dir,'Data_preprocessing.log')
file_handler = logging.FileHandler(Log_file_path)
file_handler.setLevel('DEBUG')


# setup logger formatter , assigning formatter to handlers

format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(format)
file_handler.setFormatter(format)


# finally assigning handler info to logger object
logger.addHandler(console_handler)
logger.addHandler(file_handler)


  #------------------------------------------------------------------------------------------





def preprocessing(df):

    try:
        df.dropna(axis=1)               #
        df.drop_duplicates(inplace=True)
        
        X,y = df.iloc[:,:-1],df.iloc[:,-1]
        y=LabelEncoder().fit_transform(y)

        df_num = X.select_dtypes(np.number)
        df_cat = X.select_dtypes(object)                          #

        D = {i:df_cat[i].nunique() for i in df_cat.columns}
        List = [i for i in D   if (D[i]/len(df)*100)<1.0]
                                                
        oneHOt= OneHotEncoder(sparse_output=False)
        df_cat= pd.DataFrame(oneHOt.fit_transform(df_cat[List]) , columns = oneHOt.get_feature_names_out())

        final_df = pd.concat([df_num,df_cat],axis=1)

        pca_pipeline = make_pipeline(StandardScaler(), PCA(n_components=0.99))
        final_array = pca_pipeline.fit_transform(final_df)
        logger.debug("The data has been preprocessed successfully")

        final_df = pd.concat([pd.DataFrame(final_array),pd.Series(y)], axis=1)
        return final_df
    
    except Exception as e:
        logger.error("We got somekind of error while preprocessing the data %s",e)
        raise
        

def main():
    try:
        train = pd.read_csv('/home/santosh/Desktop/MLOps/Class_3_ML_Pipeline/Data/raw/train.csv')
        test = pd.read_csv('/home/santosh/Desktop/MLOps/Class_3_ML_Pipeline/Data/raw/test.csv')

        preprocessed_train = preprocessing(train)
        preprocessed_test = preprocessing(test)

        path_prep = '/home/santosh/Desktop/MLOps/Class_3_ML_Pipeline/Data/preprocessed/'   #path_to_save_preprocessed 

        os.makedirs(path_prep,exist_ok=True)
        preprocessed_train.to_csv(os.path.join(path_prep,'train_preprocessed.csv'),index=False)
        preprocessed_test.to_csv(os.path.join(path_prep,'test_preprocessed.csv'),index=False)

        logger.debug("Data has been preprocessed and saved at: %s", path_prep)
    except Exception as e:
        logger.error("Preprocessing has failed: %s", e)
        raise


if __name__ == '__main__':
    main()

