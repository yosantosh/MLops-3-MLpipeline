import pandas as pd, numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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


logger = logging.getLogger('Data_ingetion') # creating logger object and Data_Ingetion is name of this logger object that we will gonna use it later
logger.setLevel('DEBUG')   # all log levels will be convered in debug level of logger



# setup logger handler 
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

Log_file_path = os.path.join(log_dir,'Data_ingestion.log')
file_handler = logging.FileHandler(Log_file_path)
file_handler.setLevel('DEBUG')


# setup logger formatter , assigning formatter to handlers

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


# finally assigning handler info to logger object
logger.addHandler(console_handler)
logger.addHandler(file_handler)


  #------------------------------------------------------------------------------------------


def data_loading(url):
    "Loading data usng URL and seperator = ; "
    try:
        df = pd.read_csv(url,index_col=False, sep=";")
        logger.debug("Data Loaded from: %s", url)     # debug level for step information 
        return df
    except pd.errors.ParserError as e:
        logger.error("We got error while parsing the URL data: %s", e)
        raise
    except Exception as e:
        logger.exception("We got something unknown error")     # logs traceback and error
        raise



def save_data(train:pd.DataFrame, test:pd.DataFrame,path):
    try:

        os.makedirs(path, exist_ok=True)
        train.to_csv(os.path.join(path,'train.csv'),index=False)
        test.to_csv(os.path.join(path,'test.csv'),index=False)
        logger.debug("Data has been saved to %s", path)
    except Exception as e:
        logger.exception("Failed to save data!")
        raise


def main(url,save_path,test_size=0.2):
    try:
        df = data_loading(url)
        train,test = train_test_split(df,test_size=test_size,shuffle=True,random_state=4)

        save_data(train,test,save_path)
        logger.debug('Data Ingestion has been completed successfully')
    except Exception as e:
        logger.exception("Data Ingestion failed !")

    
if __name__=='__main__':
    main(
        "https://raw.githubusercontent.com/yosantosh/MLOps/refs/heads/master/Class_3_ML_Pipeline/bank-full.csv",
        './Data/raw',
    )
