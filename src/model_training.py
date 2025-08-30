"""
Model training utilities for fake news detection.

This module contains classes and functions for training, evaluating,
and managing machine learning models for fake news classification.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.pipeline import Pipeline


class FakeNewsModelTrainer:
    """
    A comprehensive model trainer for fake news detection.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.vectorizer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize different models for comparison.
        
        Returns:
            Dict[str, Any]: Dictionary of initialized models
        """
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                C=1.0
            ),
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=20
            ),
            'SVM': SVC(
                kernel='linear',
                random_state=self.random_state,
                probability=True,
                C=1.0
            )
        }
        
        self.models = models
        return models
    
    def create_tfidf_vectorizer(self, 
                               max_features: int = 10000,
                               min_df: int = 2,
                               max_df: float = 0.95,
                               ngram_range: Tuple[int, int] = (1, 2)) -> TfidfVectorizer:
        """
        Create and configure TF-IDF vectorizer.
        
        Args:
            max_features (int): Maximum number of features
            min_df (int): Minimum document frequency
            max_df (float): Maximum document frequency
            ngram_range (Tuple[int, int]): Range of n-grams
        
        Returns:
            TfidfVectorizer: Configured vectorizer
        """
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words='english'
        )
        
        self.vectorizer = vectorizer
        return vectorizer
    
    def prepare_data(self, 
                    X: pd.Series, 
                    y: pd.Series, 
                    test_size: float = 0.2,
                    stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training by splitting and vectorizing.
        
        Args:
            X (pd.Series): Feature data (text)
            y (pd.Series): Target data (labels)
            test_size (float): Proportion of test set
            stratify (bool): Whether to stratify the split
        
        Returns:
            Tuple: X_train_tfidf, X_test_tfidf, y_train, y_test
        """
        # Split the data
        stratify_param = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        # Create vectorizer if not exists
        if self.vectorizer is None:
            self.create_tfidf_vectorizer()
        
        # Fit and transform the data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Store for later use
        self.X_train = X_train_tfidf
        self.X_test = X_test_tfidf
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train_tfidf, X_test_tfidf, y_train, y_test
    
    def train_models(self, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Train all initialized models.
        
        Args:
            verbose (bool): Whether to print progress
        
        Returns:
            Dict[str, Dict[str, Any]]: Training results for all models
        """
        if not self.models:
            self.initialize_models()
        
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        results = {}
        
        for name, model in self.models.items():
            if verbose:
                print(f"Training {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = None
            
            try:
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            except:
                pass
            
            # Calculate metrics
            results[name] = self.calculate_metrics(
                self.y_test, y_pred, y_pred_proba, model_name=name
            )
            results[name]['model'] = model
            results[name]['predictions'] = y_pred
            
            if verbose:
                print(f"  Accuracy: {results[name]['accuracy']:.4f}")
                print(f"  F1 Score: {results[name]['f1_score']:.4f}")
        
        self.results = results
        
        # Find best model
        self.find_best_model()
        
        return results
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         y_pred_proba: Optional[np.ndarray] = None,
                         model_name: str = "") -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray, optional): Prediction probabilities
            model_name (str): Name of the model
        
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, pos_label='FAKE'),
            'precision': precision_score(y_true, y_pred, pos_label='FAKE'),
            'recall': recall_score(y_true, y_pred, pos_label='FAKE')
        }
        
        if y_pred_proba is not None:
            try:
                # Convert labels to binary for ROC AUC
                y_true_binary = (y_true == 'FAKE').astype(int)
                metrics['roc_auc'] = roc_auc_score(y_true_binary, y_pred_proba)
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    def find_best_model(self, metric: str = 'f1_score') -> str:
        """
        Find the best performing model.
        
        Args:
            metric (str): Metric to use for comparison
        
        Returns:
            str: Name of the best model
        """
        if not self.results:
            raise ValueError("No results available. Train models first.")
        
        best_score = 0
        best_name = ""
        
        for name, result in self.results.items():
            if result[metric] > best_score:
                best_score = result[metric]
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.results[best_name]['model']
        
        return best_name
    
    def cross_validate_model(self, 
                            model_name: str, 
                            cv: int = 5,
                            scoring: str = 'accuracy') -> Dict[str, float]:
        """
        Perform cross-validation on a specific model.
        
        Args:
            model_name (str): Name of the model to cross-validate
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
        
        Returns:
            Dict[str, float]: Cross-validation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        
        # Combine train and test for cross-validation
        if self.X_train is not None and self.X_test is not None:
            from scipy.sparse import vstack
            X_combined = vstack([self.X_train, self.X_test])
            y_combined = np.concatenate([self.y_train, self.y_test])
        else:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        scores = cross_val_score(model, X_combined, y_combined, cv=cv, scoring=scoring)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
    
    def hyperparameter_tuning(self, 
                             model_name: str,
                             param_grid: Dict[str, List[Any]],
                             cv: int = 5,
                             scoring: str = 'f1') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_name (str): Name of the model
            param_grid (Dict[str, List[Any]]): Parameter grid
            cv (int): Cross-validation folds
            scoring (str): Scoring metric
        
        Returns:
            Dict[str, Any]: Best parameters and score
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=cv, scoring=scoring,
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 12)) -> None:
        """
        Plot comprehensive results visualization.
        
        Args:
            figsize (Tuple[int, int]): Figure size
        """
        if not self.results:
            raise ValueError("No results to plot. Train models first.")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Model comparison
        metrics_df = pd.DataFrame({
            name: {
                'Accuracy': result['accuracy'],
                'F1 Score': result['f1_score'],
                'Precision': result['precision'],
                'Recall': result['recall']
            }
            for name, result in self.results.items()
        }).T
        
        metrics_df.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Model Performance Comparison')
        axes[0,0].set_ylabel('Score')
        axes[0,0].legend()
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Confusion Matrix for best model
        if self.best_model_name:
            cm = confusion_matrix(
                self.y_test, 
                self.results[self.best_model_name]['predictions']
            )
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
            axes[0,1].set_title(f'Confusion Matrix - {self.best_model_name}')
            axes[0,1].set_xlabel('Predicted')
            axes[0,1].set_ylabel('Actual')
        
        # 3. Feature importance (if available)
        if hasattr(self.best_model, 'coef_'):
            feature_names = self.vectorizer.get_feature_names_out()
            coefficients = self.best_model.coef_[0]
            top_features = np.argsort(np.abs(coefficients))[-20:]
            
            feature_importance = pd.DataFrame({
                'feature': feature_names[top_features],
                'importance': coefficients[top_features]
            }).sort_values('importance')
            
            axes[1,0].barh(range(len(feature_importance)), feature_importance['importance'])
            axes[1,0].set_yticks(range(len(feature_importance)))
            axes[1,0].set_yticklabels(feature_importance['feature'])
            axes[1,0].set_title('Top 20 Most Important Features')
            axes[1,0].set_xlabel('Coefficient Value')
        
        # 4. ROC Curve (if probabilities available)
        try:
            for name, result in self.results.items():
                if 'roc_auc' in result and result['roc_auc'] is not None:
                    model = result['model']
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                    y_true_binary = (self.y_test == 'FAKE').astype(int)
                    
                    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba)
                    axes[1,1].plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})")
            
            axes[1,1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[1,1].set_xlabel('False Positive Rate')
            axes[1,1].set_ylabel('True Positive Rate')
            axes[1,1].set_title('ROC Curves')
            axes[1,1].legend()
        except:
            axes[1,1].text(0.5, 0.5, 'ROC curves not available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, 
                  model_path: str = "models/fake_news_model.pkl",
                  vectorizer_path: str = "models/tfidf_vectorizer.pkl") -> bool:
        """
        Save the best model and vectorizer.
        
        Args:
            model_path (str): Path to save the model
            vectorizer_path (str): Path to save the vectorizer
        
        Returns:
            bool: True if successful
        """
        if self.best_model is None or self.vectorizer is None:
            raise ValueError("No trained model or vectorizer to save.")
        
        try:
            # Create directory if it doesn't exist
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model and vectorizer
            joblib.dump(self.best_model, model_path)
            joblib.dump(self.vectorizer, vectorizer_path)
            
            print(f"Model saved to: {model_path}")
            print(f"Vectorizer saved to: {vectorizer_path}")
            
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, 
                  model_path: str = "models/fake_news_model.pkl",
                  vectorizer_path: str = "models/tfidf_vectorizer.pkl") -> bool:
        """
        Load a saved model and vectorizer.
        
        Args:
            model_path (str): Path to the saved model
            vectorizer_path (str): Path to the saved vectorizer
        
        Returns:
            bool: True if successful
        """
        try:
            self.best_model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            
            print("Model and vectorizer loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Make predictions on new texts.
        
        Args:
            texts (List[str]): List of texts to predict
        
        Returns:
            List[Tuple[str, float]]: List of (prediction, confidence) tuples
        """
        if self.best_model is None or self.vectorizer is None:
            raise ValueError("No trained model available. Train or load a model first.")
        
        # Vectorize texts
        X_vectorized = self.vectorizer.transform(texts)
        
        # Make predictions
        predictions = self.best_model.predict(X_vectorized)
        confidences = self.best_model.predict_proba(X_vectorized)
        
        results = []
        for i, pred in enumerate(predictions):
            if pred == 'FAKE':
                conf = confidences[i][0]  # Probability of FAKE
            else:
                conf = confidences[i][1]  # Probability of REAL
            
            results.append((pred, conf))
        
        return results


if __name__ == "__main__":
    # Example usage
    print("FakeNewsModelTrainer initialized successfully!")
    
    # Create sample data
    sample_texts = [
        "Breaking news: Scientists discover amazing cure",
        "Government announces new policy changes",
        "Aliens spotted in downtown area yesterday"
    ]
    
    sample_labels = ["FAKE", "REAL", "FAKE"]
    
    print("Sample texts and labels created for testing.")
