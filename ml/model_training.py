"""
NAMASTE-ICD11 Model Training Script

Standalone script to train and evaluate the TF-IDF model for 
NAMASTE to ICD-11 code mapping.

This script can be run independently to:
1. Load datasets
2. Train TF-IDF model
3. Evaluate model performance
4. Save model for use by the Flask service
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Model training class for NAMASTE-ICD11 mapping.
    
    Handles data loading, preprocessing, model training, and evaluation.
    """
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.datasets_path = self.base_path.parent / 'datasets'
        self.models_path = self.base_path / 'models'
        
        # Create models directory
        self.models_path.mkdir(exist_ok=True)
        
        # Data containers
        self.namaste_data = None
        self.icd11_data = None
        self.mappings_data = None
        
        # Model components
        self.vectorizer = None
        self.icd11_vectors = None
    
    def load_datasets(self):
        """Load all CSV datasets."""
        logger.info("🔄 Loading datasets...")
        
        # Load NAMASTE data
        namaste_path = self.datasets_path / 'namaste_100_dataset.csv'
        try:
            if namaste_path.exists():
                self.namaste_data = pd.read_csv(namaste_path)
                logger.info(f"✅ Loaded {len(self.namaste_data)} NAMASTE entries")
            else:
                logger.error(f"❌ NAMASTE dataset not found: {namaste_path}")
                return False
        except Exception as e:
            logger.error(f"❌ Error loading NAMASTE data: {str(e)}")
            return False
        
        # Load ICD-11 data
        icd11_path = self.datasets_path / 'icd11_100_dataset.csv'
        try:
            if icd11_path.exists():
                self.icd11_data = pd.read_csv(icd11_path)
                logger.info(f"✅ Loaded {len(self.icd11_data)} ICD-11 entries")
            else:
                logger.error(f"❌ ICD-11 dataset not found: {icd11_path}")
                return False
        except Exception as e:
            logger.error(f"❌ Error loading ICD-11 data: {str(e)}")
            return False
        
        # Load mappings data (for evaluation)
        mappings_path = self.datasets_path / 'namaste_icd11_100_mapping.csv'
        try:
            if mappings_path.exists():
                self.mappings_data = pd.read_csv(mappings_path)
                logger.info(f"✅ Loaded {len(self.mappings_data)} mapping entries")
            else:
                logger.warning(f"⚠️  Mappings dataset not found: {mappings_path}")
                self.mappings_data = pd.DataFrame()
        except Exception as e:
            logger.warning(f"⚠️  Error loading mappings data: {str(e)}")
            self.mappings_data = pd.DataFrame()
        
        return True
    
    def preprocess_data(self):
        """Preprocess text data for training."""
        logger.info("🔄 Preprocessing data...")
        
        # Clean NAMASTE data
        self.namaste_data['English_Translation'] = self.namaste_data['English_Translation'].fillna('')
        self.namaste_data['Category'] = self.namaste_data['Category'].fillna('')
        
        # Clean ICD-11 data
        self.icd11_data['Title'] = self.icd11_data['Title'].fillna('')
        self.icd11_data['Category'] = self.icd11_data['Category'].fillna('')
        
        logger.info("✅ Data preprocessing completed")
    
    def create_training_corpus(self):
        """
        Create training corpus combining ICD-11 and NAMASTE texts.
        
        Returns:
            list: Combined text corpus for TF-IDF training
        """
        logger.info("🔄 Creating training corpus...")
        
        corpus = []
        
        # Add ICD-11 texts
        for _, row in self.icd11_data.iterrows():
            text_parts = []
            if row['Title']:
                text_parts.append(row['Title'])
            if row['Category']:
                text_parts.append(row['Category'])
            
            if text_parts:
                corpus.append(' '.join(text_parts))
        
        # Add NAMASTE texts for better semantic understanding
        for _, row in self.namaste_data.iterrows():
            text_parts = []
            if row['English_Translation']:
                text_parts.append(row['English_Translation'])
            if row['Category']:
                text_parts.append(row['Category'])
            
            if text_parts:
                corpus.append(' '.join(text_parts))
        
        logger.info(f"✅ Created corpus with {len(corpus)} documents")
        return corpus
    
    def train_tfidf_model(self):
        """Train TF-IDF vectorizer and create ICD-11 vectors."""
        logger.info("🔄 Training TF-IDF model...")
        
        # Create corpus
        corpus = self.create_training_corpus()
        
        if not corpus:
            logger.error("❌ No text data available for training")
            return False
        
        try:
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,        # Limit vocabulary size
                stop_words='english',     # Remove common English stop words
                ngram_range=(1, 2),       # Use unigrams and bigrams
                min_df=1,                 # Minimum document frequency
                max_df=0.95,              # Maximum document frequency (remove very common words)
                sublinear_tf=True,        # Apply sublinear tf scaling
                norm='l2'                 # L2 normalization
            )
            
            # Fit vectorizer on entire corpus
            self.vectorizer.fit(corpus)
            
            # Create vectors specifically for ICD-11 entries
            icd11_texts = []
            for _, row in self.icd11_data.iterrows():
                text_parts = []
                if row['Title']:
                    text_parts.append(row['Title'])
                if row['Category']:
                    text_parts.append(row['Category'])
                
                icd11_texts.append(' '.join(text_parts) if text_parts else '')
            
            # Transform ICD-11 texts to vectors
            self.icd11_vectors = self.vectorizer.transform(icd11_texts)
            
            logger.info(f"✅ TF-IDF model trained successfully")
            logger.info(f"   • Vocabulary size: {len(self.vectorizer.vocabulary_)}")
            logger.info(f"   • ICD-11 vectors shape: {self.icd11_vectors.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training TF-IDF model: {str(e)}")
            return False
    
    def evaluate_model(self):
        """Evaluate model using existing mappings as ground truth."""
        logger.info("🔄 Evaluating model performance...")
        
        if self.mappings_data.empty:
            logger.warning("⚠️  No mappings data available for evaluation")
            return
        
        try:
            correct_predictions = 0
            total_predictions = 0
            similarity_scores = []
            
            for _, mapping in self.mappings_data.iterrows():
                # Get NAMASTE entry
                namaste_entry = self.namaste_data[
                    self.namaste_data['NAMASTE_Code'] == mapping['NAMASTE_Code']
                ].iloc[0]
                
                # Get ground truth ICD codes
                true_icd_codes = [
                    mapping['ICD11_TM2_Code'],
                    mapping['ICD11_Biomedicine_Code']
                ]
                
                # Predict using model
                query_text = namaste_entry['English_Translation']
                predictions = self._predict_for_text(query_text, top_k=5)
                
                if predictions:
                    predicted_codes = [p['icd_code'] for p in predictions]
                    
                    # Check if any true code is in top predictions
                    if any(true_code in predicted_codes for true_code in true_icd_codes):
                        correct_predictions += 1
                    
                    # Record similarity score of top prediction
                    similarity_scores.append(predictions[0]['score'])
                
                total_predictions += 1
            
            # Calculate metrics
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
            
            logger.info("📊 Model Evaluation Results:")
            logger.info(f"   • Top-5 Accuracy: {accuracy:.2%}")
            logger.info(f"   • Average Similarity Score: {avg_similarity:.3f}")
            logger.info(f"   • Total Test Cases: {total_predictions}")
            logger.info(f"   • Correct Predictions: {correct_predictions}")
            
        except Exception as e:
            logger.error(f"❌ Error during evaluation: {str(e)}")
    
    def _predict_for_text(self, text, top_k=5):
        """Helper method to predict ICD codes for text."""
        try:
            if not text.strip():
                return []
            
            # Transform query text
            query_vector = self.vectorizer.transform([text])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.icd11_vectors).flatten()
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Build results
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only positive similarities
                    icd_entry = self.icd11_data.iloc[idx]
                    results.append({
                        'icd_code': icd_entry['ICD11_Code'],
                        'title': icd_entry['Title'],
                        'score': similarities[idx]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            return []
    
    def save_model(self):
        """Save trained model components."""
        logger.info("🔄 Saving model...")
        
        try:
            # Save vectorizer
            vectorizer_path = self.models_path / 'tfidf_vectorizer.pkl'
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save ICD-11 vectors
            vectors_path = self.models_path / 'icd11_vectors.pkl'
            with open(vectors_path, 'wb') as f:
                pickle.dump(self.icd11_vectors, f)
            
            logger.info(f"✅ Model saved to {self.models_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {str(e)}")
            return False
        
        return True
    
    def run_training_pipeline(self):
        """Run complete training pipeline."""
        logger.info("🚀 Starting model training pipeline...")
        
        # Step 1: Load datasets
        if not self.load_datasets():
            logger.error("❌ Failed to load datasets. Aborting training.")
            return False
        
        # Step 2: Preprocess data
        self.preprocess_data()
        
        # Step 3: Train model
        if not self.train_tfidf_model():
            logger.error("❌ Failed to train model. Aborting.")
            return False
        
        # Step 4: Evaluate model
        self.evaluate_model()
        
        # Step 5: Save model
        if not self.save_model():
            logger.error("❌ Failed to save model.")
            return False
        
        logger.info("🎉 Training pipeline completed successfully!")
        return True

def main():
    """Main function to run training."""
    trainer = ModelTrainer()
    
    # Run training
    success = trainer.run_training_pipeline()
    
    if success:
        logger.info("✅ Model training completed successfully!")
        logger.info("   The model is now ready for use by the Flask ML service.")
    else:
        logger.error("❌ Model training failed!")
        exit(1)

if __name__ == '__main__':
    main()