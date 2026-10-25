import os
import re
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("preprocessing_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s -%(name)s-%(levelname)s-%(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

for resource_path, resource_name in [('corpora/wordnet', 'wordnet'), ('corpora/stopwords', 'stopwords'), ('corpora/omw-1.4', 'omw-1.4')]:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        try:
            nltk.download(resource_name, quiet=True)
        except Exception as e:
            logger.warning(f"Could not download {resource_name}: {e}")


def preprocess_comment(comment):
    try:
        if not isinstance(comment, str):
            comment = str(comment) if pd.notna(comment) else ""

        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', "", comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = " ".join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = " ".join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return str(comment) if comment is not None else ""


def normalize_text(df):
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug("Text normalization completed")
        return df
    except Exception as e:
        logger.error(f"Error during text normalization : {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")

        os.makedirs(interim_data_path, exist_ok=True)
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)

        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise


def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        logger.debug("starting data preprocessing...")
        root_dir = get_root_directory()
        raw_data_dir = os.path.join(root_dir, 'data', 'raw')

        train_data = pd.read_csv(os.path.join(raw_data_dir, 'train.csv'))
        test_data = pd.read_csv(os.path.join(raw_data_dir, 'test.csv'))
        logger.debug("Data loaded successfully")

        train_preprocessed_data = normalize_text(train_data)
        test_preprocessed_data = normalize_text(test_data)

        save_data(train_preprocessed_data, test_preprocessed_data, data_path=os.path.join(root_dir, 'data'))
    except Exception as e:
        logger.error("Failed to complete the data preprocessing process: %s", e)
        print(f"Error : {e}")


if __name__ == "__main__":
    main()