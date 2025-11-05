import os 
import logging
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                    #Logger setup

logger = logging.getLogger("Model evaluation")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

path = '/home/santosh/Desktop/MLOps/Class_3_ML_Pipeline/Log'
file_handler = logging.FileHandler( os.path.join(path, 'model_evaluation.log') )
file_handler.setLevel('DEBUG')

format = logging.Formatter('%(asctime)s   |  %(name)s  |  %(levelname)s  |  %(message)s')

console_handler.setFormatter(format)
file_handler.setFormatter(format)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

                           # end logger setup


def model_evaluation(dir:os.listdir, test_path,save_path:str) -> pd.DataFrame: 
    "Func to save model performance metrics"

    try:
        df = pd.read_csv(test_path)
        X_test, y_true = df.iloc[:,:-1], df.iloc[:,-1]
        logger.debug("Loaded test file")
    except Exception as e:
        logger.error('Faild to load test file with the error : %s',e)
        raise

    try:
        list_dir = os.listdir(dir)
        D={}
        for i in list_dir:
            model = joblib.load(os.path.join(dir,i))
            y_pred = model.predict(X_test)

            D[i.split('.')[0]]={'accuracy':accuracy_score(y_true,y_pred),'precision':precision_score(y_true,y_pred),'recall':recall_score(y_true,y_pred),'F1':f1_score(y_true,y_pred)}
            logger.debug('Model performance metrices has been extracted for : %s',i)


        save_df = pd.DataFrame(D)
        save_df.to_csv( os.path.join(save_path,'model_performance_metrices.csv') )
        logger.debug('Please find the Model performance data frame at: %s',save_path)
    except Exception as e:
        logger.error('Something went wrong: %s ', e)
        raise
    


def main():

    models_dir = '/home/santosh/Desktop/MLOps/Class_3_ML_Pipeline/Data/models'
    test_path = '/home/santosh/Desktop/MLOps/Class_3_ML_Pipeline/Data/preprocessed/test_preprocessed.csv'
    save_path = '/home/santosh/Desktop/MLOps/Class_3_ML_Pipeline/Data'
    
    model_evaluation(models_dir, test_path, save_path)



if __name__ == '__main__':
    main()        