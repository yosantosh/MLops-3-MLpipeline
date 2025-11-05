from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

import os
import logging
#------------------------------------------------------------------------------------------

log_dir = '/home/santosh/Desktop/MLOps/Class_3_ML_Pipeline/Log'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('Model_training')    #created  logger object
logger.setLevel('DEBUG')


console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler(os.path.join(log_dir, "Model_training.log"))
file_handler.setLevel('DEBUG')

format = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s| %(message)s")

console_handler.setFormatter(format)
file_handler.setFormatter(format)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#------------------------------------------------------------------------------------------------------------------------

models = [
    LogisticRegression(),
    SVC(kernel='rbf'),
    RandomForestClassifier(n_estimators=130,min_samples_split=5),
    KNeighborsClassifier(n_neighbors=6),
    XGBClassifier(n_estimators=130)
    ]


def model_training(L:list):

    try:
        for i in Models:
            i.fit