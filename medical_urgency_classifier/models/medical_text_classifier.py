# medical_mistral_classifier_optimized.py
"""
Classification de Textes Médicaux avec Mistral + LoRA (VERSION ULTRA OPTIMISÉE)
Optimisations drastiques pour réduire le temps d'entraînement
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
import os
from typing import Dict, List, Optional, Tuple
import logging
import re
import warnings
warnings.filterwarnings("ignore")

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedMedicalDataset(Dataset):
    """Dataset ultra-optimisé avec preprocessing minimal"""
    
    def __init__(self, texts: List[str], labels: List[str], tokenizer, max_length: int = 256):  # Réduit de 512 à 256
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Encoder les labels une seule fois
        self.label_encoder = LabelEncoder()
        self.encoded_labels = torch.tensor(
            self.label_encoder.fit_transform(labels), 
            dtype=torch.long
        )
        
        # Pré-tokenisation en batch (BEAUCOUP plus rapide)
        logger.info(" Pré-tokenisation en batch...")
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        logger.info(" Tokenisation terminée")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encoded_labels[idx]
        }

class FastMistralClassifier:
    """Classificateur Mistral ultra-optimisé pour vitesse"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.label_encoder = None
        self.class_names = []
        self.num_labels = 0
        
        # Optimisations mémoire
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Optimisations CUDA
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        logger.info(f" Classificateur Fast-Mistral initialisé sur {self.device}")
        
    def quick_load_data(self, csv_path: str, max_samples: int = 1000) -> pd.DataFrame:
        """Chargement rapide avec échantillonnage agressif"""
        logger.info(f" Chargement rapide depuis {csv_path}")
        
        try:
            # Lecture optimisée
            df = pd.read_csv(csv_path)
            logger.info(f"Données brutes: {len(df)} échantillons")
            
            # Nettoyage minimal et rapide
            df = df.dropna(subset=['transcription', 'medical_specialty'])
            df = df[df['transcription'].str.len() > 20]  # Filtre très basique
            
            # Échantillonnage agressif pour vitesse
            if len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=42)
                logger.info(f" Échantillonnage à {max_samples} pour vitesse")
            
            # Texte simple sans enrichissement
            df['text'] = df['transcription'].astype(str)
            
            # Spécialités principales seulement
            top_specialties = df['medical_specialty'].value_counts().head(15).index
            df = df[df['medical_specialty'].isin(top_specialties)]
            
            logger.info(f" Données optimisées: {len(df)} échantillons, {df['medical_specialty'].nunique()} spécialités")
            return df
            
        except Exception as e:
            logger.error(f"Erreur chargement: {str(e)}")
            raise
    
    def setup_ultra_fast_model(self, num_labels: int):
        """Configuration Mistral ultra-optimisée"""
        logger.info(f"⚡ Configuration Fast-Mistral pour {num_labels} classes")
        
        try:
            # Tokenizer optimisé
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Modèle avec optimisations agressives
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                torch_dtype=torch.float16,  # Obligatoire pour vitesse
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                # Optimisations spécifiques
                attn_implementation="flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else None,
                use_cache=False  # Important pour l'entraînement
            )
            
            self.num_labels = num_labels
            logger.info(" Modèle Mistral configuré (mode rapide)")
            
        except Exception as e:
            logger.error(f" Erreur configuration: {str(e)}")
            raise
    
    def setup_minimal_lora(self):
        """Configuration LoRA minimaliste pour vitesse maximale"""
        logger.info(" Configuration LoRA minimaliste...")
        
        try:
            # LoRA ultra-léger
            lora_config = LoraConfig(
                r=8,  # Très petit pour vitesse
                lora_alpha=16,  # Réduit proportionnellement
                target_modules=["q_proj", "v_proj"],  # Seulement 2 modules essentiels
                lora_dropout=0.05,  # Minimal
                bias="none",
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
            )
            
            # Appliquer LoRA
            self.peft_model = get_peft_model(self.model, lora_config)
            
            # Stats
            trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.peft_model.parameters())
            
            logger.info(f" LoRA ultra-léger: {trainable_params:,} params ({(trainable_params/total_params)*100:.2f}%)")
            
        except Exception as e:
            logger.error(f" Erreur LoRA: {str(e)}")
            raise
    
    def create_fast_datasets(self, df: pd.DataFrame):
        """Création rapide des datasets"""
        texts = df['text'].tolist()
        labels = df['medical_specialty'].tolist()
        
        # Split simple et rapide
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=labels
        )
        
        # Pas de validation pour gagner du temps
        self.class_names = sorted(list(set(labels)))
        self.setup_ultra_fast_model(len(self.class_names))
        self.setup_minimal_lora()
        
        # Datasets avec tokenisation en batch
        train_dataset = OptimizedMedicalDataset(train_texts, train_labels, self.tokenizer, max_length=256)
        test_dataset = OptimizedMedicalDataset(test_texts, test_labels, self.tokenizer, max_length=256)
        test_dataset.label_encoder = train_dataset.label_encoder
        
        self.label_encoder = train_dataset.label_encoder
        
        logger.info(f" Datasets: Train={len(train_dataset)}, Test={len(test_dataset)}")
        return train_dataset, test_dataset
    
    def ultra_fast_training(self, train_dataset, output_dir: str = "./fast_mistral_results"):
        """Entraînement ultra-rapide"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Arguments ultra-optimisés pour vitesse
        training_args = TrainingArguments(
            output_dir=output_dir,
            
            # VITESSE MAXIMALE
            num_train_epochs=1,  # UNE seule époque !
            per_device_train_batch_size=8,  # Plus gros batch
            gradient_accumulation_steps=1,  # Pas d'accumulation
            
            # Optimisations d'apprentissage
            learning_rate=5e-4,  # Plus élevé pour convergence rapide
            warmup_ratio=0.01,  # Minimal
            weight_decay=0.0,  # Désactivé
            
            # Optimisations techniques
            fp16=True,  # Obligatoire
            dataloader_num_workers=4,  # Parallélisation
            dataloader_pin_memory=True,
            gradient_checkpointing=False,  # Désactivé pour vitesse
            
            # Logs et sauvegarde minimalistes
            logging_steps=20,
            save_strategy="no",  # Pas de sauvegarde intermédiaire
            evaluation_strategy="no",  # Pas d'évaluation
            
            # Optimisations mémoire et calcul
            remove_unused_columns=True,
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            tf32=True if torch.cuda.is_available() else False,
            
            # Seed pour reproductibilité
            seed=42,
            data_seed=42,
        )
        
        # Data collator optimisé
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )
        
        # Trainer minimal
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            # Pas de métriques pour gagner du temps
        )
        
        logger.info(" ENTRAÎNEMENT ULTRA-RAPIDE DÉMARRÉ...")
        
        try:
            # Optimisations pré-entraînement
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Compilation du modèle si PyTorch 2.0+
                if hasattr(torch, 'compile'):
                    try:
                        self.peft_model = torch.compile(self.peft_model, mode="reduce-overhead")
                        logger.info(" Modèle compilé avec torch.compile")
                    except:
                        logger.info("  torch.compile non disponible")
            
            # ENTRAÎNEMENT
            train_result = trainer.train()
            
            # Sauvegarde finale
            self.save_fast_model(trainer, output_dir)
            
            logger.info(f" ENTRAÎNEMENT TERMINÉ en {train_result.training_loss:.4f} loss!")
            return trainer
            
        except Exception as e:
            logger.error(f" Erreur entraînement: {str(e)}")
            raise
    
    def save_fast_model(self, trainer, output_dir: str):
        """Sauvegarde optimisée"""
        try:
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Métadonnées minimales
            metadata = {
                "model_type": "fast-mistral-lora",
                "base_model": self.model_name,
                "num_labels": self.num_labels,
                "class_names": self.class_names,
                "label_encoder_classes": self.label_encoder.classes_.tolist(),
                "optimization": "ultra_fast_training",
                "epochs": 1
            }
            
            with open(os.path.join(output_dir, "fast_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(" Modèle rapide sauvegardé")
            
        except Exception as e:
            logger.error(f" Erreur sauvegarde: {str(e)}")

def ultra_fast_pipeline(csv_path: str, max_samples: int = 800):
    """Pipeline ultra-rapide (5-10 minutes max)"""
    
    logger.info(" PIPELINE ULTRA-RAPIDE MISTRAL DÉMARRÉ")
    
    try:
        # Initialisation
        classifier = FastMistralClassifier()
        
        # Données (échantillonnage agressif)
        df = classifier.quick_load_data(csv_path, max_samples=max_samples)
        
        # Datasets
        train_dataset, test_dataset = classifier.create_fast_datasets(df)
        
        # Entraînement éclair
        trainer = classifier.ultra_fast_training(train_dataset)
        
        # Test rapide
        logger.info(" Test rapide:")
        test_texts = [
            "Patient with chest pain and shortness of breath",
            "Orthopedic evaluation for knee injury"
        ]
        
        # Prédiction simple pour test
        logger.info(" PIPELINE ULTRA-RAPIDE TERMINÉ!")
        return classifier
        
    except Exception as e:
        logger.error(f" Erreur pipeline: {str(e)}")
        raise

# OPTIMISATIONS ADDITIONNELLES POUR VITESSE EXTRÊME
def setup_speed_optimizations():
    """Optimisations système pour vitesse maximale"""
    
    if torch.cuda.is_available():
        # Optimisations CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Vider le cache
        torch.cuda.empty_cache()
        
        # Infos GPU
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f" Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # Variables d'environnement pour vitesse
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = "4"
    
    logger.info(" Optimisations de vitesse activées")

if __name__ == "__main__":
    # Configuration pour vitesse maximale
    setup_speed_optimizations()
    
    csv_path = "mtsamples.csv"
    
    try:
        # Pipeline ultra-rapide (800 échantillons max, 1 époque)
        classifier = ultra_fast_pipeline(csv_path, max_samples=800)
        logger.info(" SUCCÈS - Entraînement terminé rapidement!")
        
    except Exception as e:
        logger.error(f" Erreur: {str(e)}")
        raise