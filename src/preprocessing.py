"""
Text preprocessing utilities for fake news detection.

This module contains functions for cleaning and preprocessing text data
for machine learning models.
"""

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from typing import List, Optional


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for fake news detection.
    """
    
    def __init__(self, download_nltk: bool = True):
        """
        Initialize the TextPreprocessor.
        
        Args:
            download_nltk (bool): Whether to download required NLTK data
        """
        self.stemmer = PorterStemmer()
        self.stop_words = set()
        
        if download_nltk:
            self.download_nltk_requirements()
        
        # Load stopwords after downloading
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            print("Warning: Could not load stopwords. Using empty set.")
    
    def download_nltk_requirements(self) -> bool:
        """
        Download required NLTK data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            print("NLTK requirements downloaded successfully.")
            return True
        except Exception as e:
            print(f"Error downloading NLTK data: {e}")
            return False
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and normalizing.
        
        Args:
            text (str): Input text to clean
        
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters, numbers, and extra whitespace
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text to tokenize
        
        Returns:
            List[str]: List of tokens
        """
        try:
            tokens = word_tokenize(text)
            return tokens
        except:
            # Fallback to simple split if NLTK tokenizer fails
            return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens (List[str]): List of tokens
        
        Returns:
            List[str]: Filtered tokens without stopwords
        """
        return [word for word in tokens if word not in self.stop_words and len(word) > 2]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to tokens.
        
        Args:
            tokens (List[str]): List of tokens
        
        Returns:
            List[str]: Stemmed tokens
        """
        return [self.stemmer.stem(word) for word in tokens]
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text (str): Input text to preprocess
        
        Returns:
            str: Preprocessed text
        """
        # Clean the text
        text = self.clean_text(text)
        
        if not text:
            return ""
        
        # Tokenize
        tokens = self.tokenize_text(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Apply stemming
        tokens = self.stem_tokens(tokens)
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def preprocess_dataset(self, df: pd.DataFrame, 
                          text_columns: List[str],
                          combine_columns: bool = True,
                          output_column: str = 'processed_text') -> pd.DataFrame:
        """
        Preprocess an entire dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_columns (List[str]): Columns containing text to preprocess
            combine_columns (bool): Whether to combine all text columns
            output_column (str): Name of output column
        
        Returns:
            pd.DataFrame: Dataframe with preprocessed text
        """
        df_processed = df.copy()
        
        if combine_columns:
            # Combine all text columns
            combined_text = df_processed[text_columns].fillna('').agg(' '.join, axis=1)
            df_processed[output_column] = combined_text.apply(self.preprocess_text)
        else:
            # Process each column separately
            for col in text_columns:
                processed_col = f"{col}_processed"
                df_processed[processed_col] = df_processed[col].fillna('').apply(self.preprocess_text)
        
        return df_processed
    
    def get_text_statistics(self, text: str) -> dict:
        """
        Get basic statistics about text.
        
        Args:
            text (str): Input text
        
        Returns:
            dict: Dictionary containing text statistics
        """
        if not isinstance(text, str):
            text = str(text)
        
        words = text.split()
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
        }


# Convenience functions for backward compatibility
def download_nltk_requirements():
    """Download required NLTK data."""
    preprocessor = TextPreprocessor(download_nltk=False)
    return preprocessor.download_nltk_requirements()


def clean_text(text: str) -> str:
    """Clean text using default settings."""
    preprocessor = TextPreprocessor(download_nltk=False)
    return preprocessor.clean_text(text)


def preprocess_text(text: str) -> str:
    """Preprocess text using default settings."""
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_text(text)


if __name__ == "__main__":
    # Example usage
    sample_text = """
    BREAKING NEWS: Scientists have discovered that drinking coffee 
    increases productivity by 200%! Visit www.fakenews.com for more details.
    Email us at fake@news.com for exclusive updates!!!
    """
    
    print("Original text:")
    print(sample_text)
    
    preprocessor = TextPreprocessor()
    processed = preprocessor.preprocess_text(sample_text)
    
    print("\nProcessed text:")
    print(processed)
    
    print("\nText statistics:")
    stats = preprocessor.get_text_statistics(sample_text)
    for key, value in stats.items():
        print(f"{key}: {value}")
