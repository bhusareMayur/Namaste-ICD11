"""
NAMASTE-ICD11 ML Microservice

Flask-based microservice for providing machine learning-powered 
ICD-11 code suggestions based on NAMASTE traditional medicine terms.

Uses TF-IDF vectorization and cosine similarity for semantic matching.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODEL_PATH = Path(__file__).parent / 'models'
DATASETS_PATH = Path(__file__).parent.parent / 'datasets'

class MLMappingService:
    """
    Machine Learning service for NAMASTE to ICD-11 mapping suggestions.
    
    This service:
    1. Loads NAMASTE and ICD-11 datasets from CSV files
    2. Creates TF-IDF vectors from text descriptions
    3. Provides similarity-based ICD-11 suggestions for NAMASTE terms
    """
    
    def __init__(self):
        self.vectorizer = None
        self.icd11_vectors = None
        self.icd11_data = None
        self.namaste_data = None
        self.is_trained = False
        
        # Create models directory if it doesn't exist
        MODEL_PATH.mkdir(exist_ok=True)
        
        # Initialize the service
        self._load_data()
        self._train_or_load_model()
    
    def _load_data(self):
        """Load NAMASTE and ICD-11 datasets from CSV files."""
        try:
            # Load NAMASTE data
            namaste_path = DATASETS_PATH / 'namaste_100_dataset.csv'
            if namaste_path.exists():
                self.namaste_data = pd.read_csv(namaste_path)
                logger.info(f"✅ Loaded {len(self.namaste_data)} NAMASTE entries")
            else:
                logger.warning(f"⚠️  NAMASTE dataset not found: {namaste_path}")
                self.namaste_data = pd.DataFrame()
            
            # Load ICD-11 data
            icd11_path = DATASETS_PATH / 'icd11_100_dataset.csv'
            if icd11_path.exists():
                self.icd11_data = pd.read_csv(icd11_path)
                logger.info(f"✅ Loaded {len(self.icd11_data)} ICD-11 entries")
            else:
                logger.warning(f"⚠️  ICD-11 dataset not found: {icd11_path}")
                self.icd11_data = pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error loading datasets: {str(e)}")
            # Create empty DataFrames as fallback
            self.namaste_data = pd.DataFrame(columns=['NAMASTE_Code', 'Traditional_Term', 'English_Translation', 'Medical_System', 'Category'])
            self.icd11_data = pd.DataFrame(columns=['ICD11_Code', 'Title', 'Module', 'Category', 'Code_Type'])
    
    def _create_text_corpus(self):
        """
        Create text corpus for TF-IDF training.
        
        Combines:
        - ICD-11 titles and categories
        - NAMASTE English translations and categories
        """
        corpus = []
        
        # Add ICD-11 text data
        if not self.icd11_data.empty:
            for _, row in self.icd11_data.iterrows():
                text_parts = []
                if pd.notna(row.get('Title')):
                    text_parts.append(str(row['Title']))
                if pd.notna(row.get('Category')):
                    text_parts.append(str(row['Category']))
                
                corpus.append(' '.join(text_parts))
        
        # Add NAMASTE text data for better semantic understanding
        if not self.namaste_data.empty:
            for _, row in self.namaste_data.iterrows():
                text_parts = []
                if pd.notna(row.get('English_Translation')):
                    text_parts.append(str(row['English_Translation']))
                if pd.notna(row.get('Category')):
                    text_parts.append(str(row['Category']))
                
                corpus.append(' '.join(text_parts))
        
        return corpus
    
    def _train_or_load_model(self):
        """Train TF-IDF model or load existing one."""
        vectorizer_path = MODEL_PATH / 'tfidf_vectorizer.pkl'
        vectors_path = MODEL_PATH / 'icd11_vectors.pkl'
        
        try:
            # Try to load existing model
            if vectorizer_path.exists() and vectors_path.exists():
                logger.info("🔄 Loading existing TF-IDF model...")
                
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                with open(vectors_path, 'rb') as f:
                    self.icd11_vectors = pickle.load(f)
                
                logger.info("✅ Loaded existing TF-IDF model successfully")
                self.is_trained = True
                return
        
        except Exception as e:
            logger.warning(f"⚠️  Could not load existing model: {str(e)}")
        
        # Train new model
        self._train_model()
    
    def _train_model(self):
        """Train TF-IDF model from scratch."""
        logger.info("🔄 Training new TF-IDF model...")
        
        try:
            # Create text corpus
            corpus = self._create_text_corpus()
            
            if not corpus:
                logger.warning("⚠️  No text data available for training")
                # Create dummy model
                self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                self.vectorizer.fit(['no data available'])
                self.icd11_vectors = np.array([])
                self.is_trained = False
                return
            
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,  # Limit features for efficiency
                stop_words='english',
                ngram_range=(1, 2),  # Include unigrams and bigrams
                min_df=1,  # Minimum document frequency
                max_df=0.95  # Maximum document frequency
            )
            
            # Fit vectorizer on full corpus
            self.vectorizer.fit(corpus)
            
            # Create vectors for ICD-11 entries only
            if not self.icd11_data.empty:
                icd11_texts = []
                for _, row in self.icd11_data.iterrows():
                    text_parts = []
                    if pd.notna(row.get('Title')):
                        text_parts.append(str(row['Title']))
                    if pd.notna(row.get('Category')):
                        text_parts.append(str(row['Category']))
                    
                    icd11_texts.append(' '.join(text_parts))
                
                self.icd11_vectors = self.vectorizer.transform(icd11_texts)
            else:
                self.icd11_vectors = np.array([])
            
            # Save model
            self._save_model()
            
            logger.info(f"✅ TF-IDF model trained successfully with {len(corpus)} documents")
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"❌ Error training model: {str(e)}")
            self.is_trained = False
    
    def _save_model(self):
        """Save trained model to disk."""
        try:
            vectorizer_path = MODEL_PATH / 'tfidf_vectorizer.pkl'
            vectors_path = MODEL_PATH / 'icd11_vectors.pkl'
            
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            with open(vectors_path, 'wb') as f:
                pickle.dump(self.icd11_vectors, f)
            
            logger.info("✅ Model saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {str(e)}")
    
    def predict_icd11_codes(self, text, top_k=5):
        """
        Predict ICD-11 codes for given text using cosine similarity.
        
        Args:
            text (str): Input text to match against ICD-11 codes
            top_k (int): Number of top suggestions to return
            
        Returns:
            list: List of dictionaries with ICD code, title, and score
        """
        if not self.is_trained or self.icd11_vectors is None or len(self.icd11_vectors.shape) == 1:
            logger.warning("⚠️  Model not trained or no ICD-11 data available")
            return []
        
        try:
            # Transform input text
            text_vector = self.vectorizer.transform([text])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(text_vector, self.icd11_vectors).flatten()
            
            # Get top-k similar entries
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.icd11_data) and similarities[idx] > 0:  # Only include positive similarities
                    icd_entry = self.icd11_data.iloc[idx]
                    results.append({
                        'icd_code': str(icd_entry['ICD11_Code']),
                        'title': str(icd_entry['Title']),
                        'score': float(similarities[idx]),
                        'module': str(icd_entry.get('Module', 'Unknown')),
                        'category': str(icd_entry.get('Category', 'Unknown'))
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in prediction: {str(e)}")
            return []

# Initialize ML service
try:
    ml_service = MLMappingService()
    logger.info("🚀 ML Mapping Service initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize ML service: {str(e)}")
    ml_service = None

# Flask routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = {
        'status': 'ok' if ml_service and ml_service.is_trained else 'degraded',
        'service': 'namaste-icd11-ml',
        'version': '1.0.0',
        'model_trained': ml_service.is_trained if ml_service else False,
        'icd11_entries': len(ml_service.icd11_data) if ml_service and ml_service.icd11_data is not None else 0,
        'namaste_entries': len(ml_service.namaste_data) if ml_service and ml_service.namaste_data is not None else 0
    }
    
    return jsonify(status)

@app.route('/predict', methods=['GET'])
def predict():
    """
    Predict ICD-11 codes for given text.
    
    Query Parameters:
        text (str): Text to analyze and find similar ICD-11 codes
        top_k (int, optional): Number of top suggestions (default: 5)
    
    Returns:
        JSON array of top ICD-11 code suggestions with scores
    """
    # Get query parameters
    text = request.args.get('text', '').strip()
    top_k = request.args.get('top_k', 5)
    
    # Validate input
    if not text:
        return jsonify({
            'error': 'Text parameter is required',
            'usage': '/predict?text=fever&top_k=5'
        }), 400
    
    try:
        top_k = int(top_k)
        if top_k < 1 or top_k > 20:
            top_k = 5
    except ValueError:
        top_k = 5
    
    # Check if service is available
    if not ml_service or not ml_service.is_trained:
        return jsonify({
            'error': 'ML service is not available or not trained',
            'suggestions': [],
            'text': text,
            'note': 'Please ensure datasets are loaded and model is trained'
        }), 503
    
    # Get predictions
    try:
        suggestions = ml_service.predict_icd11_codes(text, top_k)
        
        return jsonify(suggestions)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal prediction error',
            'suggestions': [],
            'text': text,
            'message': 'Please try again later'
        }), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """
    Retrain the ML model with current data.
    
    This endpoint allows retraining the model if new data is added.
    """
    if not ml_service:
        return jsonify({
            'error': 'ML service not available'
        }), 503
    
    try:
        # Reload data and retrain
        ml_service._load_data()
        ml_service._train_model()
        
        return jsonify({
            'message': 'Model retrained successfully',
            'model_trained': ml_service.is_trained,
            'icd11_entries': len(ml_service.icd11_data) if ml_service.icd11_data is not None else 0,
            'namaste_entries': len(ml_service.namaste_data) if ml_service.namaste_data is not None else 0
        })
        
    except Exception as e:
        logger.error(f"❌ Retrain error: {str(e)}")
        return jsonify({
            'error': 'Retraining failed',
            'message': str(e)
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with service information."""
    return jsonify({
        'service': 'NAMASTE-ICD11 ML Microservice',
        'version': '1.0.0',
        'description': 'Machine learning service for traditional medicine to ICD-11 code mapping',
        'endpoints': {
            '/health': 'Service health check',
            '/predict?text=<term>': 'Get ICD-11 suggestions for text',
            '/retrain': 'Retrain the ML model (POST)'
        },
        'status': 'ok' if ml_service and ml_service.is_trained else 'not_ready'
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/health', '/predict', '/retrain', '/']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'Please try again later'
    }), 500

if __name__ == '__main__':
    # Configuration
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Starting NAMASTE-ICD11 ML Service on port {port}")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )