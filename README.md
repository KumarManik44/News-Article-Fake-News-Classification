# 📰 Fake News Detection System

A machine learning project that classifies news articles as fake or real using Natural Language Processing techniques and Logistic Regression.

## 🎯 Project Overview

This project implements an end-to-end fake news detection system with a web interface. It achieves **92.19% accuracy** on a balanced dataset of 6,335 news articles using TF-IDF vectorization and Logistic Regression.

## 🚀 Live Demo

[**Try the App Here**](https://fakenewsclassification84532.streamlit.app/)

## 🚀 Features

- **High Accuracy**: 92.19% accuracy with 92.34% F1-score
- **Balanced Dataset**: 50-50 split between fake and real news
- **Web Interface**: Interactive Streamlit application
- **Text Preprocessing**: Comprehensive NLP pipeline
- **Model Comparison**: Logistic Regression vs Naive Bayes
- **Cross Validation**: Robust model evaluation

## 📊 Model Performance

| Metric | Logistic Regression | Naive Bayes |
|--------|-------------------|-------------|
| Accuracy | **92.19%** | 88.16% |
| F1-Score | **92.34%** | 88.51% |
| Precision (Fake) | 90% | 86% |
| Recall (Fake) | 94% | 91% |

## 🛠️ Technical Stack

- **Python 3.8+**
- **Scikit-learn**: Machine learning models
- **NLTK**: Natural language processing
- **Pandas & NumPy**: Data manipulation
- **Streamlit**: Web interface
- **Matplotlib & Seaborn**: Data visualization

## 📁 Project Structure

```
fake-news-detection/
├── data/
│   └── news.csv                    # Dataset
├── models/
│   ├── fake_news_model.pkl         # Trained model
│   └── tfidf_vectorizer.pkl        # TF-IDF vectorizer
├── notebooks/
│   └── Fake_News_Detection.ipynb   # Complete analysis
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Text preprocessing functions
│   └── model_training.py           # Model training utilities
├── app/
│   └── fake_news_app.py            # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
```

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
```

### 4. Run the Web Application
```bash
streamlit run app/fake_news_app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage

### Web Interface
1. Open the Streamlit application
2. Choose to either type/paste your own text or select an example
3. Click "Analyze Article" 
4. View the prediction results with confidence scores

### Jupyter Notebook
Run the complete analysis notebook:
```bash
jupyter notebook notebooks/Fake_News_Detection.ipynb
```

## 🧪 Dataset

The project uses a Kaggle dataset containing 6,335 news articles with the following features:
- **Title**: Article headline
- **Text**: Article content  
- **Label**: FAKE or REAL classification

Dataset is perfectly balanced with 50% fake and 50% real news articles.

## 🔧 Methodology

### 1. Data Preprocessing
- Text cleaning (remove special characters, convert to lowercase)
- Tokenization using NLTK
- Stopword removal
- Stemming with Porter Stemmer
- Combine title and text for richer features

### 2. Feature Engineering
- TF-IDF Vectorization
- Unigrams and bigrams (1-2 word combinations)
- Maximum 10,000 features
- Min document frequency: 2
- Max document frequency: 95%

### 3. Model Training
- Train-test split: 80-20
- Models compared: Logistic Regression, Naive Bayes
- Cross-validation for robust evaluation
- Hyperparameter tuning

### 4. Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix analysis
- 5-fold Cross-validation
- Feature importance analysis

## 📈 Results & Insights

### Key Findings:
- **Logistic Regression** outperformed Naive Bayes
- **TF-IDF with bigrams** captured important phrase patterns
- **Cross-validation** confirmed model stability (91.20% ± 0.84%)
- **Feature analysis** revealed most discriminative terms

### Model Strengths:
- High accuracy on balanced dataset
- Fast prediction time
- Interpretable results
- Robust preprocessing pipeline

## 🔮 Future Improvements

- [ ] Implement advanced models (Random Forest, XGBoost, BERT)
- [ ] Add word embeddings (Word2Vec, GloVe)
- [ ] Implement ensemble methods
- [ ] Add explainability features (LIME/SHAP)
- [ ] Real-time news article scraping
- [ ] Multi-language support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- NLTK library for natural language processing
- Scikit-learn for machine learning algorithms
- Streamlit for the web interface

---

⭐ **Star this repository if you found it helpful!**
