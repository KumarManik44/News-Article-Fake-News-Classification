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
        return True
    except Exception as e:
        st.error(f"Failed to download NLTK data: {str(e)}")
        return False

# Load the trained model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/fake_news_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.info("Please ensure 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl' are in the 'models' directory")
        return None, None

# Text preprocessing functions
def clean_text(text):
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def preprocess_text(text):
    """Complete preprocessing pipeline"""
    try:
        text = clean_text(text)
        
        if not text:
            return ""
        
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        
        return ' '.join(tokens)
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        return text  # Return original text if preprocessing fails

# Prediction function
def predict_news(text, model, vectorizer):
    """Predict if news is fake or real"""
    try:
        processed_text = preprocess_text(text)
        
        if not processed_text:
            return "ERROR", 0.0, ""
        
        text_vector = vectorizer.transform([processed_text])
        prediction = model.predict(text_vector)[0]
        confidence = model.predict_proba(text_vector)[0]
        
        if prediction == 'FAKE':
            conf_score = confidence[0]
        else:
            conf_score = confidence[1]
        
        return prediction, conf_score, processed_text
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return "ERROR", 0.0, ""

# Example articles
def get_example_articles():
    """Return example articles for testing"""
    return {
        "Real News Example": """
        The Federal Reserve announced today that it will maintain current interest rates at 0.25% 
        following their monthly meeting. Fed Chair Jerome Powell cited ongoing economic uncertainty 
        and inflation concerns as primary factors in this decision. The announcement came after 
        extensive deliberation among board members and consultation with leading economists. 
        Market analysts had predicted this outcome based on recent economic indicators and 
        employment data released last week.
        """.strip(),
        
        "Fake News Example": """
        BREAKING: Scientists have discovered that aliens built the pyramids using advanced 
        anti-gravity technology. A secret government document leaked yesterday reveals that 
        extraterrestrial beings visited Earth 4,000 years ago and helped ancient Egyptians 
        construct these monuments. The document also claims that similar structures exist on 
        Mars and can be seen with special telescopes. Government officials refuse to comment 
        on this explosive revelation.
        """.strip(),
        
        "Satirical Example": """
        Local man discovers that turning off the news actually improves his mental health, 
        shocking researchers nationwide. In a groundbreaking study of one person, scientists 
        found that avoiding 24/7 news cycles led to decreased anxiety and improved sleep. 
        "We never expected such dramatic results," said Dr. Obvious from the Institute of 
        Common Sense. The man reportedly now spends his time gardening and talking to neighbors.
        """.strip()
    }

# Main Streamlit application
def main():
    # Page configuration
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize NLTK data
    nltk_ready = download_nltk_data()
    
    # Header
    st.title("📰 Fake News Detection System")
    st.markdown("*An AI-powered tool to identify potentially fake news articles*")
    st.markdown("---")
    
    # Load model
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.error("⚠️ Model loading failed. Please check that model files are properly uploaded.")
        st.stop()
    
    if not nltk_ready:
        st.warning("⚠️ NLTK data download incomplete. Some features may not work properly.")
    
    # Sidebar information
    with st.sidebar:
        st.header("📊 Model Information")
        st.info("""
        **Algorithm:** Logistic Regression  
        **Accuracy:** 92.19%  
        **F1 Score:** 92.34%  
        **Features:** TF-IDF Vectors  
        **Training Data:** 6,335 news articles
        """)
        
        st.header("🔍 How it works")
        st.markdown("""
        1. **Text Preprocessing**: Cleans and normalizes the input text
        2. **Feature Extraction**: Converts text to TF-IDF vectors
        3. **Classification**: Uses trained model to predict fake/real
        4. **Confidence Score**: Provides prediction certainty
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This tool is for educational purposes. 
        Always verify news from multiple credible sources.
        """)
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("Enter News Article")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["✍️ Type/Paste Text", "📄 Example Articles"],
            horizontal=True
        )
        
        user_input = ""
        
        if input_method == "✍️ Type/Paste Text":
            user_input = st.text_area(
                "Paste your news article here:",
                height=250,
                placeholder="Enter the complete news article text you want to analyze...",
                help="Paste the full article including headline for best results"
            )
            
        else:  # Example Articles
            example_articles = get_example_articles()
            
            selected_example = st.selectbox(
                "Select an example article:",
                ["Choose an example..."] + list(example_articles.keys())
            )
            
            if selected_example != "Choose an example...":
                user_input = example_articles[selected_example]
                st.text_area(
                    "Selected example:",
                    value=user_input,
                    height=200,
                    disabled=True
                )
    
    with col2:
        st.header("Analysis Results")
        
        # Single analyze button with unique key
        analyze_clicked = st.button(
            "🔍 Analyze Article",
            type="primary",
            disabled=not user_input,
            use_container_width=True,
            key="analyze_button"
        )
        
        if analyze_clicked and user_input:
            with st.spinner("🔄 Analyzing article..."):
                prediction, confidence, processed_text = predict_news(user_input, model, vectorizer)
                
                if prediction == "ERROR":
                    st.error("❌ Analysis failed. Please try again with different text.")
                else:
                    # Display results
                    if prediction == "FAKE":
                        st.error("🚨 **FAKE NEWS DETECTED**")
                        st.markdown(f"**Confidence:** {confidence:.1%}")
                        
                        # Color-coded confidence bar
                        progress_color = "🔴" if confidence > 0.8 else "🟡"
                        st.progress(confidence)
                        st.caption(f"{progress_color} High confidence" if confidence > 0.8 else f"{progress_color} Moderate confidence")
                        
                    else:
                        st.success("✅ **REAL NEWS DETECTED**")
                        st.markdown(f"**Confidence:** {confidence:.1%}")
                        
                        # Color-coded confidence bar
                        progress_color = "🟢" if confidence > 0.8 else "🟡"
                        st.progress(confidence)
                        st.caption(f"{progress_color} High confidence" if confidence > 0.8 else f"{progress_color} Moderate confidence")
                    
                    # Additional metrics
                    st.markdown("### 📈 Analysis Metrics")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Original Words", len(user_input.split()))
                        st.metric("Processed Words", len(processed_text.split()) if processed_text else 0)
                    
                    with col_b:
                        st.metric("Characters", len(user_input))
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Processing details
                    with st.expander("🔍 Processing Details", expanded=False):
                        st.markdown("**Preprocessing Steps Applied:**")
                        st.markdown("• Text cleaning and normalization")
                        st.markdown("• Tokenization and stopword removal")
                        st.markdown("• Stemming and feature extraction")
                        
                        if processed_text:
                            st.markdown("**Processed Text Preview:**")
                            preview = processed_text[:300] + "..." if len(processed_text) > 300 else processed_text
                            st.code(preview, language="text")
                        
                        st.markdown("**Model Decision Factors:**")
                        st.info("The model analyzes word patterns, phrase structures, and linguistic features to make its prediction.")
        
        elif analyze_clicked and not user_input:
            st.warning("📝 Please enter or select some text to analyze!")
        
        # Tips section
        if not user_input:
            st.markdown("### 💡 Tips for Better Results")
            st.markdown("""
            - Include the complete article with headline
            - Longer articles generally yield better predictions  
            - Try the example articles to see how it works
            - The model works best with English news content
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p><strong>Fake News Detection System</strong></p>
        <p>Built with Streamlit • Powered by Machine Learning • For Educational Use</p>
        <p>⚠️ Always verify news from multiple credible sources</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
