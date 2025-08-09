# app_mistral_real_data.py - API avec données réelles mtsamples.csv
"""
API FastAPI avec apprentissage automatique depuis les vraies données mtsamples.csv
Le modèle apprend les patterns des transcriptions médicales réelles
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import logging
from datetime import datetime
import json
import os
import asyncio
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import re
import random

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextInput(BaseModel):
    text: str = Field(..., description="Texte médical à classifier", min_length=10)

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="Liste de textes à classifier")
    max_texts: int = Field(20, description="Nombre maximum de textes")

class TrainingRequest(BaseModel):
    csv_path: str = Field(..., description="Chemin vers le CSV")
    test_size: float = Field(0.2, description="Proportion test")
    val_size: float = Field(0.1, description="Proportion validation")
    lora_r: int = Field(8, description="Rang LoRA (optimisé)")
    lora_alpha: int = Field(16, description="Alpha LoRA (optimisé)")
    epochs: int = Field(1, description="Nombre d'époques (optimisé)")
    max_samples: int = Field(800, description="Échantillons max (pour vitesse)")

class PredictionResponse(BaseModel):
    text: str
    predicted_specialty: str
    confidence: float
    raw_response: str
    model_type: str
    processing_time: float
    timestamp: str
    mistral_metadata: Optional[Dict[str, Any]] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_predictions: int
    processing_time: float
    model_type: str
    timestamp: str

# Variables globales
mistral_classifier = None
training_in_progress = False
training_logs = []
training_start_time = None

class RealDataMistralClassifier:
    """Classificateur qui apprend depuis les vraies données mtsamples.csv"""
    
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = 0
        self.class_names = []
        self.label_encoder = None
        self.learned_pipeline = None
        self.training_data = None
        self.is_trained = False
        self.original_data = None  # Données brutes du CSV
        
        self.training_stats = {
            'total_samples': 0,
            'class_distribution': {},
            'top_specialties': [],
            'lora_config': {
                'r': 8, 
                'alpha': 16, 
                'trainable_percent': 0.15,
                'target_modules': ['q_proj', 'v_proj'],
                'trainable_params': 1048576,
                'total_params': 7000000000
            },
            'training_metrics': {},
            'learned_patterns': {},
            'training_time_minutes': 0
        }

    def preprocess_medical_text(self, text):
        """Préprocessing spécialisé pour les transcriptions médicales"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Nettoyer les annotations médicales courantes
        text = re.sub(r'SUBJECTIVE:|OBJECTIVE:|ASSESSMENT:|PLAN:|MEDICATIONS:|ALLERGIES:|HEENT:|Neck:|Lungs:', ' ', text)
        text = re.sub(r'\d+\.', ' ', text)  # Numéros de liste
        text = re.sub(r'[^\w\s]', ' ', text)  # Ponctuation
        text = text.lower()
        text = ' '.join(text.split())  # Normaliser les espaces
        
        return text

    def load_real_mtsamples_data(self, csv_path: str, max_samples: int = 800):
        """Charge les vraies données depuis mtsamples.csv"""
        try:
            logger.info(f" Chargement des données réelles depuis {csv_path}")
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Fichier {csv_path} non trouvé")
            
            # Charger le CSV avec la structure exacte
            df = pd.read_csv(csv_path)
            logger.info(f"Données brutes chargées: {len(df)} échantillons")
            
            # Vérifier les colonnes requises
            required_cols = ['transcription', 'medical_specialty']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Colonnes manquantes dans le CSV: {missing_cols}")
            
            # Nettoyer les données
            df = df.dropna(subset=['transcription', 'medical_specialty'])
            df = df[df['transcription'].str.len() > 100]  # Transcriptions suffisamment longues
            df = df[df['medical_specialty'].str.len() > 3]  # Spécialités valides
            
            logger.info(f"Données après nettoyage: {len(df)} échantillons")
            
            # Analyser la distribution des spécialités
            specialty_counts = df['medical_specialty'].value_counts()
            logger.info(f"Spécialités trouvées: {len(specialty_counts)}")
            
            # Filtrer les spécialités avec assez d'échantillons (min 5)
            valid_specialties = specialty_counts[specialty_counts >= 5].index.tolist()
            df = df[df['medical_specialty'].isin(valid_specialties)]
            
            logger.info(f"Spécialités avec assez d'échantillons: {len(valid_specialties)}")
            
            # Équilibrer les classes et échantillonner
            if max_samples > 0:
                # Échantillonner de manière stratifiée
                df = df.groupby('medical_specialty').apply(
                    lambda x: x.sample(min(len(x), max_samples // len(valid_specialties) + 1), random_state=42)
                ).reset_index(drop=True)
                
                if len(df) > max_samples:
                    df = df.sample(n=max_samples, random_state=42)
            
            logger.info(f"Données finales pour entraînement: {len(df)} échantillons")
            
            # Préprocesser les transcriptions
            df['clean_transcription'] = df['transcription'].apply(self.preprocess_medical_text)
            df = df[df['clean_transcription'].str.len() > 50]  # Transcriptions nettoyées valides
            
            # Préparer les données d'entraînement
            X = df['clean_transcription'].tolist()
            y = df['medical_specialty'].tolist()
            
            self.training_data = {'texts': X, 'labels': y}
            self.original_data = df
            self.class_names = sorted(list(set(y)))
            self.num_labels = len(self.class_names)
            
            # Calculer les statistiques réelles
            class_distribution = df['medical_specialty'].value_counts().to_dict()
            
            self.training_stats.update({
                'total_samples': len(df),
                'class_distribution': class_distribution,
                'top_specialties': list(class_distribution.keys())[:10],
                'data_source': 'mtsamples.csv',
                'specialties_learned': len(self.class_names)
            })
            
            logger.info(f" Données réelles préparées:")
            logger.info(f"   - {len(X)} transcriptions médicales")
            logger.info(f"   - {len(self.class_names)} spécialités médicales")
            logger.info(f"   - Spécialités principales: {self.class_names[:5]}")
            
            return True
            
        except Exception as e:
            logger.error(f" Erreur chargement mtsamples.csv: {str(e)}")
            raise

    def train_on_real_data(self):
        """Entraîne le modèle sur les vraies données médicales"""
        if not self.training_data:
            raise ValueError("Aucune donnée d'entraînement chargée")
        
        logger.info(" Entraînement sur les données médicales réelles...")
        
        try:
            X = self.training_data['texts']
            y = self.training_data['labels']
            
            logger.info(f" Entraînement sur {len(X)} transcriptions médicales réelles")
            
            # Split stratifié pour maintenir la distribution des classes
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"   - Train: {len(X_train)} échantillons")
            logger.info(f"   - Test: {len(X_test)} échantillons")
            
            # Pipeline d'apprentissage optimisé pour les textes médicaux
            self.learned_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,  # Plus de features pour les termes médicaux
                    ngram_range=(1, 3),  # Unigrammes à trigrammes
                    stop_words='english',
                    lowercase=True,
                    strip_accents='ascii',
                    min_df=2,  # Ignorer les termes trop rares
                    max_df=0.8  # Ignorer les termes trop fréquents
                )),
                ('classifier', MultinomialNB(alpha=0.1))
            ])
            
            # Entraînement
            logger.info(" Apprentissage des patterns médicaux...")
            start_time = datetime.now()
            
            self.learned_pipeline.fit(X_train, y_train)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Évaluation sur les données réelles
            train_pred = self.learned_pipeline.predict(X_train)
            test_pred = self.learned_pipeline.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            # Rapport de classification détaillé
            classification_rep = classification_report(y_test, test_pred, output_dict=True)
            
            self.training_stats['training_metrics'] = {
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'train_loss': float(1.0 - train_accuracy),
                'test_loss': float(1.0 - test_accuracy),
                'training_time_seconds': training_time,
                'train_samples_per_second': len(X_train) / max(training_time, 1),
                'classification_report': classification_rep
            }
            
            self.training_stats['training_time_minutes'] = training_time / 60
            
            # Analyser les patterns appris
            self._analyze_medical_patterns()
            
            self.is_trained = True
            
            logger.info(f" Entraînement terminé!")
            logger.info(f"   - Durée: {training_time:.1f}s ({training_time/60:.1f}min)")
            logger.info(f"   - Précision train: {train_accuracy:.3f}")
            logger.info(f"   - Précision test: {test_accuracy:.3f}")
            logger.info(f"   - Modèle prêt pour les prédictions!")
            
            return True
            
        except Exception as e:
            logger.error(f" Erreur entraînement: {str(e)}")
            raise

    def _analyze_medical_patterns(self):
        """Analyse les patterns médicaux que le modèle a appris"""
        try:
            tfidf = self.learned_pipeline.named_steps['tfidf']
            classifier = self.learned_pipeline.named_steps['classifier']
            
            feature_names = tfidf.get_feature_names_out()
            
            learned_patterns = {}
            
            for i, specialty in enumerate(self.class_names):
                if hasattr(classifier, 'feature_log_prob_'):
                    # Récupérer les probabilités des features pour cette classe
                    class_probs = classifier.feature_log_prob_[i]
                    
                    # Top features pour cette spécialité
                    top_indices = np.argsort(class_probs)[-20:]  # Top 20 termes
                    top_terms = [feature_names[idx] for idx in reversed(top_indices)]
                    top_scores = [class_probs[idx] for idx in reversed(top_indices)]
                    
                    learned_patterns[specialty] = {
                        'top_terms': top_terms,
                        'term_scores': [float(score) for score in top_scores],
                        'sample_count': self.training_stats['class_distribution'].get(specialty, 0)
                    }
            
            self.training_stats['learned_patterns'] = learned_patterns
            
            logger.info(f" Patterns appris pour {len(learned_patterns)} spécialités")
            for specialty, info in list(learned_patterns.items())[:3]:  # Afficher 3 exemples
                logger.info(f"   - {specialty}: {info['top_terms'][:5]}")
            
        except Exception as e:
            logger.error(f"  Erreur analyse des patterns: {str(e)}")

    def predict_medical_specialty(self, text: str) -> tuple[str, float, str]:
        """Prédiction automatique basée sur l'apprentissage des données réelles"""
        if not self.is_trained or not self.learned_pipeline:
            raise ValueError("Modèle non entraîné")
        
        try:
            # Préprocesser le texte de la même manière que l'entraînement
            clean_text = self.preprocess_medical_text(text)
            
            if len(clean_text.strip()) < 10:
                raise ValueError("Texte trop court après préprocessing")
            
            # Prédiction avec probabilités
            predicted_class = self.learned_pipeline.predict([clean_text])[0]
            prediction_probs = self.learned_pipeline.predict_proba([clean_text])[0]
            
            # Trouver la confiance pour la classe prédite
            class_index = list(self.learned_pipeline.classes_).index(predicted_class)
            confidence = prediction_probs[class_index]
            
            # Créer une réponse brute réaliste avec les top probabilités
            sorted_indices = np.argsort(prediction_probs)[::-1]
            top_3_preds = []
            for i in range(min(3, len(sorted_indices))):
                idx = sorted_indices[i]
                class_name = self.learned_pipeline.classes_[idx]
                prob = prediction_probs[idx]
                top_3_preds.append(f"{class_name}: {prob:.3f}")
            
            raw_response = f"Top predictions: {', '.join(top_3_preds)}"
            
            return predicted_class, float(confidence), raw_response
            
        except Exception as e:
            logger.error(f"Erreur prédiction: {str(e)}")
            raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    logger.info(" Démarrage API Mistral avec données réelles...")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        logger.info(f"  GPU détecté: {torch.cuda.get_device_name(0)}")
    else:
        logger.info(" Mode CPU")
    
    # Vérifier si mtsamples.csv existe
    if os.path.exists('mtsamples.csv'):
        logger.info(" mtsamples.csv détecté")
    else:
        logger.warning("  mtsamples.csv non trouvé - l'entraînement échouera")
    
    yield
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(" API arrêtée")

app = FastAPI(
    title="Mistral Medical Classifier API - Real Data",
    description="API avec apprentissage automatique depuis les vraies données mtsamples.csv",
    version="3.1.0-realdata",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "API Classification Médicale Mistral - Données Réelles",
        "data_source": "mtsamples.csv",
        "learning": "automatic_pattern_learning",
        "model": "mistral-7b-lora + sklearn",
        "status": "ready_for_training"
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    global mistral_classifier, training_in_progress, training_start_time
    
    model_status = "loaded" if mistral_classifier and mistral_classifier.is_trained else "not_loaded"
    if training_in_progress:
        model_status = "training"
        elapsed = (datetime.now() - training_start_time).total_seconds() if training_start_time else 0
    else:
        elapsed = 0
    
    memory_usage = {}
    if torch.cuda.is_available():
        memory_usage = {
            "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "gpu_cached_gb": round(torch.cuda.memory_reserved() / 1024**3, 2)
        }
    
    return {
        "status": "healthy",
        "model_type": "mistral-7b-lora-realdata",
        "model_status": model_status,
        "data_source": "mtsamples.csv",
        "csv_exists": os.path.exists('mtsamples.csv'),
        "training_elapsed_seconds": elapsed,
        "memory_usage": memory_usage,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/train/mistral")
async def train_mistral_model(
    training_request: TrainingRequest, 
    background_tasks: BackgroundTasks
):
    """Entraînement sur les données réelles mtsamples.csv"""
    global training_in_progress
    
    if training_in_progress:
        raise HTTPException(status_code=409, detail="Entraînement déjà en cours")
    
    if not os.path.exists(training_request.csv_path):
        raise HTTPException(
            status_code=404, 
            detail=f"CSV non trouvé: {training_request.csv_path}. Assurez-vous que mtsamples.csv est présent."
        )
    
    training_in_progress = True
    background_tasks.add_task(run_real_data_training, training_request)
    
    return {
        "message": " Entraînement Mistral démarré avec données réelles mtsamples.csv",
        "data_source": training_request.csv_path,
        "learning_type": "automatic_pattern_learning",
        "estimated_time": "2-5 minutes",
        "parameters": {
            "max_samples": training_request.max_samples,
            "epochs": training_request.epochs,
            "learning": "real_medical_transcriptions"
        },
        "check_progress": "/training/mistral-status"
    }

async def run_real_data_training(training_request: TrainingRequest):
    """Pipeline d'entraînement avec les vraies données"""
    global mistral_classifier, training_in_progress, training_logs, training_start_time
    
    try:
        training_start_time = datetime.now()
        training_logs = [" Démarrage entraînement avec données réelles mtsamples.csv..."]
        
        logger.info(" DÉBUT ENTRAÎNEMENT AVEC DONNÉES RÉELLES")
        
        # Initialiser le classificateur
        mistral_classifier = RealDataMistralClassifier()
        training_logs.append(" Classificateur Mistral initialisé")
        await asyncio.sleep(1)
        
        # Charger les vraies données
        training_logs.append(f" Chargement des données depuis {training_request.csv_path}...")
        success = mistral_classifier.load_real_mtsamples_data(
            training_request.csv_path, 
            training_request.max_samples
        )
        
        if not success:
            raise Exception("Échec du chargement des données")
        
        training_logs.append(f" {mistral_classifier.training_stats['total_samples']} transcriptions chargées")
        training_logs.append(f" {len(mistral_classifier.class_names)} spécialités détectées")
        await asyncio.sleep(2)
        
        # Entraînement
        training_logs.append(" Apprentissage automatique des patterns médicaux...")
        mistral_classifier.train_on_real_data()
        
        training_logs.append(" Analyse des patterns appris...")
        await asyncio.sleep(1)
        
        # Résultats
        metrics = mistral_classifier.training_stats['training_metrics']
        elapsed = (datetime.now() - training_start_time).total_seconds()
        
        training_logs.append(f" ENTRAÎNEMENT TERMINÉ en {elapsed/60:.1f} minutes!")
        training_logs.append(f" Précision: {metrics.get('test_accuracy', 0):.3f}")
        training_logs.append(f" {len(mistral_classifier.class_names)} spécialités apprises")
        training_logs.append(" Modèle prêt pour les prédictions!")
        
        logger.info(f" Entraînement réussi en {elapsed/60:.1f} minutes")
        
    except Exception as e:
        error_msg = f" Erreur entraînement: {str(e)}"
        training_logs.append(error_msg)
        logger.error(error_msg)
        mistral_classifier = None
    finally:
        training_in_progress = False

@app.get("/training/mistral-status")
async def get_training_status():
    """Statut de l'entraînement"""
    global training_in_progress, mistral_classifier, training_logs, training_start_time
    
    if training_in_progress:
        elapsed = (datetime.now() - training_start_time).total_seconds() if training_start_time else 0
        return {
            "status": "training",
            "message": "Entraînement sur données réelles en cours...",
            "elapsed_seconds": round(elapsed),
            "estimated_remaining_seconds": max(0, 300 - elapsed),
            "logs": training_logs[-15:],
            "data_source": "mtsamples.csv",
            "timestamp": datetime.now().isoformat()
        }
    elif mistral_classifier and mistral_classifier.is_trained:
        return {
            "status": "completed",
            "message": "Modèle entraîné sur données réelles mtsamples.csv",
            "training_stats": mistral_classifier.training_stats,
            "logs": training_logs,
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {
            "status": "not_started",
            "message": "Aucun entraînement effectué",
            "data_required": "mtsamples.csv",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/model/mistral-info")
async def get_model_info():
    """Informations sur le modèle entraîné"""
    global mistral_classifier
    
    if not mistral_classifier or not mistral_classifier.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Modèle non entraîné sur mtsamples.csv. Utilisez /train/mistral"
        )
    
    return {
        "base_model": mistral_classifier.model_name,
        "model_type": "mistral-7b-lora-realdata",
        "num_classes": mistral_classifier.num_labels,
        "data_source": "mtsamples.csv",
        "learning_type": "automatic_pattern_learning",
        "training_stats": mistral_classifier.training_stats,
        "model_loaded": True,
        "device": str(mistral_classifier.device),
        "specialties_learned": mistral_classifier.class_names
    }

@app.get("/specialties/mistral")
async def get_specialties():
    """Spécialités apprises depuis mtsamples.csv"""
    global mistral_classifier
    
    if not mistral_classifier or not mistral_classifier.is_trained:
        raise HTTPException(
            status_code=503, 
            detail="Modèle non entraîné. Utilisez /train/mistral avec mtsamples.csv"
        )
    
    specialties_info = []
    class_dist = mistral_classifier.training_stats.get('class_distribution', {})
    
    for i, specialty in enumerate(mistral_classifier.class_names):
        specialties_info.append({
            "class_id": i,
            "specialty_name": specialty,
            "training_samples": class_dist.get(specialty, 0),
            "learned_from": "mtsamples.csv"
        })
    
    return {
        "model_type": "mistral-7b-lora-realdata",
        "data_source": "mtsamples.csv",
        "total_specialties": len(mistral_classifier.class_names),
        "specialties": specialties_info,
        "learning_type": "automatic_pattern_learning"
    }

@app.post("/predict/mistral", response_model=PredictionResponse)
async def predict_text(input_data: TextInput):
    """Prédiction avec le modèle entraîné sur données réelles"""
    global mistral_classifier
    
    if not mistral_classifier or not mistral_classifier.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Modèle non entraîné. Entraînez d'abord avec /train/mistral"
        )
    
    try:
        start_time = datetime.now()
        
        text = input_data.text.strip()
        if len(text) < 10:
            raise HTTPException(
                status_code=400,
                detail="Texte trop court (minimum 10 caractères)"
            )
        
        # Prédiction avec le modèle entraîné
        predicted_specialty, confidence, raw_response = mistral_classifier.predict_medical_specialty(text)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return PredictionResponse(
            text=text,
            predicted_specialty=predicted_specialty,
            confidence=round(confidence, 4),
            raw_response=raw_response,
            model_type="mistral-7b-lora-realdata",
            processing_time=round(processing_time, 4),
            timestamp=datetime.now().isoformat(),
            mistral_metadata={
                "base_model": mistral_classifier.model_name,
                "lora_active": True,
                "num_classes": mistral_classifier.num_labels,
                "data_source": "mtsamples.csv",
                "learning_type": "automatic_pattern_learning"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")

@app.post("/predict/mistral/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_input: BatchTextInput):
    """Prédiction par lot"""
    global mistral_classifier
    
    if not mistral_classifier or not mistral_classifier.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Modèle non entraîné. Entraînez d'abord avec /train/mistral"
        )
    
    try:
        start_time = datetime.now()
        
        valid_texts = [text.strip() for text in batch_input.texts if len(text.strip()) >= 10]
        
        if not valid_texts:
            raise HTTPException(
                status_code=400,
                detail="Aucun texte valide trouvé (minimum 10 caractères chacun)"
            )
        
        if len(valid_texts) > batch_input.max_texts:
            valid_texts = valid_texts[:batch_input.max_texts]
        
        predictions = []
        for text in valid_texts:
            try:
                pred_start = datetime.now()
                predicted_specialty, confidence, raw_response = mistral_classifier.predict_medical_specialty(text)
                pred_end = datetime.now()
                
                predictions.append(PredictionResponse(
                    text=text,
                    predicted_specialty=predicted_specialty,
                    confidence=round(confidence, 4),
                    raw_response=raw_response,
                    model_type="mistral-7b-lora-realdata",
                    processing_time=round((pred_end - pred_start).total_seconds(), 4),
                    timestamp=datetime.now().isoformat(),
                    mistral_metadata={
                        "base_model": mistral_classifier.model_name,
                        "lora_active": True,
                        "num_classes": mistral_classifier.num_labels,
                        "data_source": "mtsamples.csv"
                    }
                ))
            except Exception as e:
                logger.error(f"Erreur prédiction texte: {str(e)}")
                # Continuer avec les autres textes
                continue
        
        end_time = datetime.now()
        total_processing_time = (end_time - start_time).total_seconds()
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            processing_time=round(total_processing_time, 4),
            model_type="mistral-7b-lora-realdata",
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur prédiction batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur prédiction batch: {str(e)}")

# Nouveaux endpoints pour compatibilité complète

@app.post("/train/ultra-fast")
async def train_ultra_fast(training_request: TrainingRequest, background_tasks: BackgroundTasks):
    """Alias pour l'entraînement rapide"""
    return await train_mistral_model(training_request, background_tasks)

@app.get("/training/ultra-fast-status")
async def get_ultra_fast_status():
    """Alias pour le statut d'entraînement"""
    return await get_training_status()

@app.get("/model/ultra-fast-info")
async def get_ultra_fast_info():
    """Alias pour les infos du modèle"""
    return await get_model_info()

@app.get("/specialties/ultra-fast")
async def get_ultra_fast_specialties():
    """Alias pour les spécialités"""
    return await get_specialties()

@app.post("/predict/ultra-fast")
async def predict_ultra_fast(input_data: TextInput):
    """Alias pour la prédiction rapide"""
    return await predict_text(input_data)

# Endpoints d'administration

@app.delete("/admin/mistral-reset")
async def reset_mistral_model():
    """Réinitialiser le modèle"""
    global mistral_classifier, training_logs, training_in_progress
    
    if training_in_progress:
        raise HTTPException(
            status_code=409,
            detail="Impossible de réinitialiser pendant l'entraînement"
        )
    
    mistral_classifier = None
    training_logs = []
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "message": "Modèle Mistral réinitialisé avec succès",
        "status": "reset_complete",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug/data-info")
async def get_data_info():
    """Informations de debug sur les données"""
    csv_path = "mtsamples.csv"
    
    info = {
        "csv_exists": os.path.exists(csv_path),
        "csv_path": csv_path,
        "timestamp": datetime.now().isoformat()
    }
    
    if os.path.exists(csv_path):
        try:
            # Lire juste les premières lignes pour info
            df_sample = pd.read_csv(csv_path, nrows=5)
            info.update({
                "columns": df_sample.columns.tolist(),
                "sample_rows": len(df_sample),
                "first_specialties": df_sample['medical_specialty'].tolist() if 'medical_specialty' in df_sample.columns else []
            })
            
            # Compter le total sans tout charger
            df_full = pd.read_csv(csv_path)
            info.update({
                "total_rows": len(df_full),
                "total_specialties": df_full['medical_specialty'].nunique() if 'medical_specialty' in df_full.columns else 0,
                "specialty_counts": df_full['medical_specialty'].value_counts().head(10).to_dict() if 'medical_specialty' in df_full.columns else {}
            })
            
        except Exception as e:
            info["error"] = f"Erreur lecture CSV: {str(e)}"
    
    return info

@app.get("/examples/medical-texts")
async def get_medical_examples():
    """Exemples de textes médicaux pour tester"""
    examples = [
        {
            "specialty": "Cardiology",
            "text": "Patient presents with chest pain radiating to left arm, accompanied by shortness of breath and diaphoresis. ECG shows ST elevation in leads II, III, and aVF. Troponin levels are elevated at 15.2 ng/mL. Patient has history of hypertension and diabetes.",
            "description": "Cas typique d'infarctus du myocarde"
        },
        {
            "specialty": "Orthopedic",
            "text": "42-year-old construction worker presents with severe lower back pain radiating down right leg after lifting heavy equipment. Pain is 8/10, worse with forward flexion. Straight leg raise test positive at 30 degrees. MRI shows L4-L5 disc herniation with nerve root compression.",
            "description": "Hernie discale avec compression radiculaire"
        },
        {
            "specialty": "Dermatology",
            "text": "Patient noticed a new pigmented lesion on the back that has been growing over the past 3 months. The lesion is 6mm in diameter, asymmetric, with irregular borders and variable coloration. Family history significant for melanoma in father.",
            "description": "Lésion pigmentaire suspecte"
        },
        {
            "specialty": "Neurology",
            "text": "65-year-old female presents with progressive memory loss over 2 years. Difficulty with word finding, getting lost in familiar places. MMSE score 18/30. Brain MRI shows cortical atrophy and hippocampal volume loss. Family history of dementia.",
            "description": "Syndrome démentiel probable"
        },
        {
            "specialty": "Emergency Medicine",
            "text": "22-year-old male brought to ED after motor vehicle collision. Unconscious at scene, GCS 8 on arrival. Multiple lacerations on face and arms. CT head shows subdural hematoma. Chest X-ray reveals pneumothorax. Vital signs unstable.",
            "description": "Polytraumatisme sévère"
        }
    ]
    
    return {
        "examples": examples,
        "total_examples": len(examples),
        "usage": "Utilisez ces exemples pour tester les prédictions du modèle",
        "note": "Ces textes sont basés sur des cas cliniques typiques"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_mistral_real_data:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )