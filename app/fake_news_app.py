
import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

download_nltk_data()

# Load the trained model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('models/fake_news_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Prediction function
def predict_news(text, model, vectorizer):
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    confidence = model.predict_proba(text_vector)[0]
    
    if prediction == 'FAKE':
        conf_score = confidence[0]
    else:
        conf_score = confidence[1]
    
    return prediction, conf_score, processed_text

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("Fake News Detection System")
    st.markdown("---")
    
    # Load model
    try:
        model, vectorizer = load_model()
        st.success("Model loaded successfully!")
    except:
        st.error("Error loading model. Please ensure model files are in the same directory.")
        return
    
    # Sidebar with model info
    st.sidebar.header("Model Information")
    st.sidebar.info("""
    **Model:** Logistic Regression  
    **Accuracy:** 92.19%  
    **F1 Score:** 92.34%  
    **Features:** TF-IDF Vectors  
    **Training Data:** 6,335 news articles
    """)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Enter News Article")
        
        # Text input options
        input_method = st.radio("Choose input method:", ["Type/Paste Text", "Example Articles"])
        
        if input_method == "Type/Paste Text":
            user_input = st.text_area(
                "Paste your news article here:",
                height=200,
                placeholder="Enter the news article text you want to analyze..."
            )
        else:
            example_choice = st.selectbox("Select an example:", [
                "Select an example...",
                "Real News Example",
                "Fake News Example"
            ])
            
            examples = {
                "Real News Example": "The Federal Reserve announced today that it will maintain current interest rates at 0.25% following their monthly meeting. Fed Chair Jerome Powell cited ongoing economic uncertainty and inflation concerns as primary factors in this decision. The announcement came after extensive deliberation among board members and consultation with leading economists.",
                "Fake News Example": "BREAKING: Scientists have discovered that aliens built the pyramids using advanced anti-gravity technology. A secret government document leaked yesterday reveals that extraterrestrial beings visited Earth 4,000 years ago and helped ancient Egyptians construct these monuments. The document also claims that similar structures exist on Mars."
            }
            
            if example_choice != "Select an example...":
                user_input = examples[example_choice]
                st.text_area("Selected example:", value=user_input, height=150, disabled=True)
            else:
                user_input = ""
    
    with col2:
        st.header("Analysis Results")
        
        if st.button("Analyze Article", type="primary") and user_input:
            with st.spinner("Analyzing..."):
                try:
                    prediction, confidence, processed_text = predict_news(user_input, model, vectorizer)
                    
                    # Display results
                    if prediction == "FAKE":
                        st.error(f"**FAKE NEWS DETECTED**")
                        st.metric("Confidence", f"{confidence:.1%}")
                    else:
                        st.success(f"**REAL NEWS DETECTED**")
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Confidence bar
                    st.progress(confidence)
                    
                    # Additional info
                    with st.expander("Processing Details"):
                        st.write("**Original text length:**", len(user_input.split()), "words")
                        st.write("**Processed text length:**", len(processed_text.split()), "words")
                        st.write("**Processed text preview:**")
                        st.code(processed_text[:200] + "..." if len(processed_text) > 200 else processed_text)
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        
        elif not user_input and st.button("Analyze Article", type="primary"):
            st.warning("Please enter some text to analyze!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with Streamlit • Powered by Machine Learning</p>
        <p>Model trained on news article dataset for educational purposes</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


