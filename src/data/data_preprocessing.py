import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

logger=logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler=logging.FileHandler("preprocessing_errors.log")
file_handler.setLevel("ERROR")

formatter=logging.Formatter("%(asctime)s -%(name)s-%(levelname)s-%(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_comment(comment):
    try:
        comment=comment.lower()
        
        comment=comment.strip()
        
        comment=re.sub(r'[^A-Za-z0-9\s!?.,]',"",comment)
        
        stop_words=set(stopwords.words('english'))-{'not','but','however','no','yet'}
        comment=" ".join([word for word in comment.split() if word not in stop_words])
        
        lemmatizer=WordNetLemmatizer()
        comment=" ".join([lemmatizer.lemmatize(word) for word in comment.split()])
        
        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment

def normalize_text(df):
    try:
        df['clean_comment']=df['clean_comment'].apply(preprocess_comment)
        logger.debug("Text normalization completed")
        return df
    except Exception as e:
        logger.error(f"Error during text normalization : {e}")
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
        interim_data_path=os.path.join(data_path,'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path,exist_ok=True)
        logger.debug(f"Directory {interim_data_path} created or already exists")
        
        train_data.to_csv(os.path.join(interim_data_path,"train_processed.csv"),index=False)
        test_data.to_csv(os.path.join(interim_data_path,"test_processed.csv"),index=False)

        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occured while saving data: {e}")
        raise 

def main():
    try:
        logger.debug("starting data preprocessing...")
        
        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv('./data/raw/test.csv')
        logger.debug("Data loaded successfully")
        
        train_preprocessed_data=normalize_text(train_data)
        test_preprocessed_data=normalize_text(test_data)
        
        save_data(train_preprocessed_data,test_preprocessed_data,data_path='./data')
    except Exception as e:
        logger.error("Failed to complete the data preprocessing process: %s",e)
        print(f"Error : {e}")

if __name__=="__main__":
    main()