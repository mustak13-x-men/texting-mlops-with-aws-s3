import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)

def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')
        
        # Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def ensure_raw_data(train_path='./data/raw/train.csv', test_path='./data/raw/test.csv'):
    """
    Ensure raw train/test CSV files exist, or build them from the fallback experiments/spam.csv dataset.
    """
    if os.path.exists(train_path) and os.path.exists(test_path):
        return train_path, test_path

    fallback_path = 'experiments/spam.csv'
    if os.path.exists(fallback_path):
        logger.debug('Raw files not found, building dataset from %s', fallback_path)
        df = pd.read_csv(fallback_path)

        if 'v1' in df.columns and 'v2' in df.columns:
            df = df.rename(columns={'v1': 'target', 'v2': 'text'})
            drop_cols = [c for c in ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'] if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)
        elif 'target' not in df.columns or 'text' not in df.columns:
            raise ValueError('Fallback dataset must contain v1/v2 or target/text columns.')

        train, test = train_test_split(df, test_size=0.2, random_state=42)
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        logger.debug('Fallback raw train/test files created at %s and %s', train_path, test_path)
        return train_path, test_path

    raise FileNotFoundError(
        'Missing raw data files. Expected ./data/raw/train.csv and ./data/raw/test.csv, or experiments/spam.csv to build them.'
    )


def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        train_path, test_path = ensure_raw_data()
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Store the data inside data/interim
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()