"""
Fake News Detection Package

A comprehensive machine learning package for detecting fake news using 
Natural Language Processing techniques and various classification algorithms.

This package provides:
- Text preprocessing utilities
- Model training and evaluation tools
- Pre-trained models for immediate use
- Web interface for interactive predictions

Author: Your Name
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Kumar Manik"
__email__ = "kumar2000.manik@gmail.com"
__license__ = "MIT"

# Import main classes for easy access
from .preprocessing import TextPreprocessor, preprocess_text, clean_text, download_nltk_requirements
from .model_training import FakeNewsModelTrainer

# Define what gets imported with "from src import *"
__all__ = [
    # Classes
    'TextPreprocessor',
    'FakeNewsModelTrainer',
    
    # Functions
    'preprocess_text',
    'clean_text', 
    'download_nltk_requirements',
    
    # Metadata
    '__version__',
    '__author__',
    '__email__',
    '__license__'
]

# Package-level configuration
DEFAULT_MODEL_CONFIG = {
    'tfidf_max_features': 10000,
    'tfidf_min_df': 2,
    'tfidf_max_df': 0.95,
    'tfidf_ngram_range': (1, 2),
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

# Supported models
SUPPORTED_MODELS = [
    'Logistic Regression',
    'Naive Bayes', 
    'Random Forest',
    'SVM'
]

# Package information
PACKAGE_INFO = {
    'name': 'fake-news-detection',
    'description': 'Machine Learning package for fake news detection',
    'keywords': ['machine-learning', 'nlp', 'fake-news', 'classification', 'text-analysis'],
    'requirements': [
        'pandas>=1.5.0',
        'numpy>=1.21.0', 
        'scikit-learn>=1.1.0',
        'nltk>=3.7',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'joblib>=1.2.0'
    ]
}


def get_version():
    """Get the current version of the package."""
    return __version__


def get_package_info():
    """Get comprehensive package information."""
    return {
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'license': __license__,
        'supported_models': SUPPORTED_MODELS,
        'default_config': DEFAULT_MODEL_CONFIG,
        **PACKAGE_INFO
    }


def setup_nltk_data():
    """
    Download required NLTK data for the package.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        return download_nltk_requirements()
    except ImportError:
        print("NLTK not installed. Please install with: pip install nltk")
        return False


def quick_start_example():
    """
    Print a quick start example for using the package.
    """
    example_code = '''
# Quick Start Example - Fake News Detection

# 1. Import the package
from src import TextPreprocessor, FakeNewsModelTrainer
import pandas as pd

# 2. Load your data
df = pd.read_csv('data/news.csv')

# 3. Preprocess text
preprocessor = TextPreprocessor()
df['processed_text'] = df['text'].apply(preprocessor.preprocess_text)

# 4. Train models
trainer = FakeNewsModelTrainer()
X_train, X_test, y_train, y_test = trainer.prepare_data(
    df['processed_text'], df['label']
)
results = trainer.train_models()

# 5. Make predictions
predictions = trainer.predict(['Your news article text here'])
print(f"Prediction: {predictions[0][0]}, Confidence: {predictions[0][1]:.3f}")

# 6. Save the model
trainer.save_model()
'''
    
    print("=== QUICK START EXAMPLE ===")
    print(example_code)


def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        dict: Status of each dependency
    """
    import importlib
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'sklearn': 'scikit-learn',
        'nltk': 'nltk',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'joblib': 'joblib'
    }
    
    status = {}
    
    for import_name, package_name in required_packages.items():
        try:
            importlib.import_module(import_name)
            status[package_name] = "✅ Installed"
        except ImportError:
            status[package_name] = "❌ Missing"
    
    return status


def print_dependency_status():
    """Print the status of all dependencies."""
    print("=== DEPENDENCY STATUS ===")
    status = check_dependencies()
    
    for package, stat in status.items():
        print(f"{package}: {stat}")
    
    missing = [pkg for pkg, stat in status.items() if "Missing" in stat]
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
    else:
        print("\n🎉 All dependencies are installed!")


# Package initialization
def _initialize_package():
    """Initialize the package when imported."""
    try:
        # Try to setup NLTK data silently
        setup_nltk_data()
    except:
        # Don't fail if NLTK setup fails during import
        pass


# Run initialization
_initialize_package()


# Convenience function for interactive use
def info():
    """Display package information and quick start guide."""
    print("=" * 60)
    print(f"🔍 FAKE NEWS DETECTION PACKAGE v{__version__}")
    print("=" * 60)
    
    info_dict = get_package_info()
    print(f"📧 Author: {info_dict['author']}")
    print(f"📝 Description: {info_dict['description']}")
    print(f"🔧 Supported Models: {', '.join(info_dict['supported_models'])}")
    
    print("\n" + "=" * 60)
    print_dependency_status()
    
    print("\n" + "=" * 60)
    quick_start_example()
    
    print("=" * 60)
    print("For more help, visit: https://github.com/yourusername/fake-news-detection")
    print("=" * 60)


# Welcome message (only shown in interactive environments)
if __name__ != "__main__":
    try:
        # Check if we're in an interactive environment
        import sys
        if hasattr(sys, 'ps1'):
            print(f"📰 Fake News Detection Package v{__version__} loaded successfully!")
            print("💡 Type 'info()' for package information and quick start guide.")
    except:
        pass
