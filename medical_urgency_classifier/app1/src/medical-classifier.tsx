
import React, { useState, useEffect } from 'react';
import type { ReactElement } from 'react';
import { AlertCircle, Brain, Upload, Play, BarChart3, Zap, Download, Trash2, Send, Clock, CheckCircle, Activity, FileText, Database, TrendingUp, Users, Target, Stethoscope, Settings, Cpu, MemoryStick, ChevronRight, Info, Shield, Sparkles } from 'lucide-react';

const MistralMedicalClassifierApp = () => {
  const [inputText, setInputText] = useState('');
  const [batchTexts, setBatchTexts] = useState(['']);
  
  interface MistralPrediction {
    text: string;
    predicted_specialty: string;
    confidence: number;
    raw_response: string;
    model_type: string;
    processing_time: number;
    timestamp: string;
    mistral_metadata?: {
      base_model: string;
      lora_active: boolean;
      num_classes: number;
    };
  }
  
  const [predictions, setPredictions] = useState<MistralPrediction[]>([]);
  const [loading, setLoading] = useState(false);
  
  interface ApiHealth {
    status: string;
    model_type: string;
    model_status: string;
    mistral_info?: {
      base_model: string;
      num_classes: number;
      lora_active: boolean;
      specialties_learned: number;
      training_samples: number;
    };
    memory_usage?: {
      gpu?: {
        allocated_gb: number;
        reserved_gb: number;
        device_name: string;
      };
      lora?: {
        total_params: number;
        trainable_params: number;
        lora_efficiency: string;
      };
    };
    cuda_available: boolean;
  }
  
  const [apiHealth, setApiHealth] = useState<ApiHealth | null>(null);
  
  interface MistralModelInfo {
    base_model: string;
    model_type: string;
    lora_config: {
      r: number;
      alpha: number;
      dropout: number;
      target_modules: string[];
      trainable_params: number;
      total_params: number;
      trainable_percent: number;
    };
    training_stats: {
      total_samples: number;
      class_distribution: { [key: string]: number };
      training_metrics?: {
        train_loss: number;
        train_runtime: number;
        train_samples_per_second: number;
      };
    };
    model_loaded: boolean;
    device: string;
  }
  
  const [modelInfo, setModelInfo] = useState<MistralModelInfo | null>(null);
  const [trainingStatus, setTrainingStatus] = useState('not_started');
  
  interface Specialty {
    class_id: number;
    specialty_name: string;
    training_samples: number;
    learned_by: string;
  }
  
  const [specialties, setSpecialties] = useState<Specialty[]>([]);
  const [activeTab, setActiveTab] = useState('predict');
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  
  // Configuration d'entraînement
  const [trainingConfig, setTrainingConfig] = useState({
    csv_path: 'mtsamples.csv',
    test_size: 0.2,
    val_size: 0.1,
    lora_r: 32,
    lora_alpha: 64,
    epochs: 3
  });

  const API_BASE_URL = 'http://localhost:8000';

  useEffect(() => {
    checkApiHealth();
    checkTrainingStatus();
    loadModelInfo();
    loadSpecialties();
    
    const interval = setInterval(() => {
      if (trainingStatus === 'training') {
        checkTrainingStatus();
      }
    }, 5000);
    
    return () => clearInterval(interval);
  }, [trainingStatus]);

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      setApiHealth(data);
    } catch (error) {
      console.error('Erreur de connexion API:', error);
      setApiHealth({ 
        status: 'error', 
        model_status: 'unavailable',
        model_type: 'mistral-7b-lora',
        cuda_available: false
      });
    }
  };

  const loadModelInfo = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/model/mistral-info`);
      if (response.ok) {
        const data = await response.json();
        setModelInfo(data);
      }
    } catch (error) {
      console.log('Modèle Mistral non encore entraîné');
    }
  };

  const loadSpecialties = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/specialties/mistral`);
      if (response.ok) {
        const data = await response.json();
        setSpecialties(data.specialties || []);
      }
    } catch (error) {
      console.log('Pas encore de spécialités Mistral apprises');
    }
  };

  const checkTrainingStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/training/mistral-status`);
      const data = await response.json();
      setTrainingStatus(data.status);
      setTrainingLogs(data.logs || []);
      
      if (data.status === 'completed') {
        loadModelInfo();
        loadSpecialties();
        checkApiHealth();
      }
    } catch (error) {
      console.error('Erreur status entraînement Mistral:', error);
    }
  };

  const startMistralTraining = async () => {
    setLoading(true);
    setTrainingStatus('training');

    try {
      const response = await fetch(`${API_BASE_URL}/train/mistral`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainingConfig),
      });

      const result = await response.json();
      
      if (response.ok) {
        alert(`Entraînement Mistral + LoRA démarré!\n${result.message}`);
        setTrainingLogs([result.message]);
      } else {
        setTrainingStatus('error');
        alert('Erreur entraînement Mistral: ' + result.detail);
      }
    } catch (error) {
      setTrainingStatus('error');
      if (error instanceof Error) {
        alert('Erreur: ' + error.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSinglePrediction = async () => {
    if (!inputText.trim()) return;

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/predict/mistral`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Erreur de prédiction Mistral');
      }

      const result = await response.json();
      setPredictions([result, ...predictions]);
      setInputText('');
    } catch (error) {
      if (error instanceof Error) {
        alert('Erreur lors de la prédiction Mistral: ' + error.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleBatchPrediction = async () => {
    const validTexts = batchTexts.filter(text => text.trim());
    if (validTexts.length === 0) return;

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/predict/mistral/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ texts: validTexts }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Erreur de prédiction en lot Mistral');
      }

      const results = await response.json();
      setPredictions([...results.predictions, ...predictions]);
      setBatchTexts(['']);
    } catch (error) {
      if (error instanceof Error) {
        alert('Erreur lors de la prédiction en lot Mistral: ' + error.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handlePredictionClear = () => {
    setPredictions([]);
  };

  const handleDownload = () => {
    const data = JSON.stringify(predictions, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'mistral_predictions.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
      const text = e.target?.result as string;
      const lines = text.split('\n').filter(line => line.trim() !== '');
      setBatchTexts(lines);
    };
    reader.readAsText(file);
  };

  const addBatchInput = () => {
    setBatchTexts([...batchTexts, '']);
  };

  const updateBatchInput = (index: number, value: string) => {
    const newBatchTexts = [...batchTexts];
    newBatchTexts[index] = value;
    setBatchTexts(newBatchTexts);
  };

  const removeBatchInput = (index: number) => {
    const newBatchTexts = batchTexts.filter((_, i) => i !== index);
    setBatchTexts(newBatchTexts.length > 0 ? newBatchTexts : ['']);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return 'text-emerald-600';
    if (confidence > 0.5) return 'text-amber-600';
    return 'text-red-600';
  };

  const getStatusClasses = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'status-healthy';
      case 'training':
        return 'status-training';
      case 'error':
      case 'unavailable':
        return 'status-error';
      default:
        return 'bg-gray-400 text-white';
    }
  };

  const getTabClass = (tabName: string) => {
    let baseClass = 'px-4 py-2 text-sm font-semibold rounded-lg transition-all duration-200';
    if (activeTab === tabName) {
      let activeClass = '';
      switch (tabName) {
        case 'predict':
          activeClass = 'tab-active-blue';
          break;
        case 'batch':
          activeClass = 'tab-active-purple';
          break;
        case 'train':
          activeClass = 'tab-active-green';
          break;
        case 'stats':
          activeClass = 'tab-active-orange';
          break;
        default:
          activeClass = 'tab-active-blue';
          break;
      }
      return `${baseClass} ${activeClass} text-white shadow-lg`;
    } else {
      return `${baseClass} bg-white text-gray-700 hover:bg-gray-100`;
    }
  };

  const getIconColorClass = (specialty: string) => {
    const colors = ['emerald', 'blue', 'purple', 'orange', 'green', 'red', 'indigo', 'slate', 'teal', 'cyan', 'violet', 'rose', 'amber'];
    const index = specialty.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % colors.length;
    return `icon-${colors[index]}`;
  };

  const getDotColorClass = (specialty: string) => {
    const colors = ['emerald', 'blue', 'purple', 'orange', 'green', 'red', 'indigo', 'slate', 'teal', 'cyan', 'violet', 'rose', 'amber'];
    const index = specialty.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % colors.length;
    return `dot-${colors[index]}`;
  };

  const getStatCardColor = (index: number) => {
    const colors = ['emerald', 'blue', 'purple', 'orange'];
    return `stat-card-${colors[index % colors.length]}`;
  };

  const getPredictionCardColor = (confidence: number) => {
    if (confidence > 0.8) return 'border-emerald-200 bg-emerald-50';
    if (confidence > 0.5) return 'border-amber-200 bg-amber-50';
    return 'border-red-200 bg-red-50';
  };

  const getProgressBarColor = (index: number) => {
    const colors = ['emerald', 'blue', 'purple', 'indigo'];
    return `progress-${colors[index % colors.length]}`;
  };

  const getConfidenceBadgeClass = (confidence: number) => {
    if (confidence > 0.8) return 'confidence-high';
    if (confidence > 0.5) return 'confidence-medium';
    return 'confidence-low';
  };
  
  const trainingButtonClass = trainingStatus === 'training'
    ? 'disabled:bg-gray-400 bg-blue-600'
    : 'btn-emerald';

  const predictButtonClass = loading
    ? 'disabled:bg-gray-400 bg-blue-600'
    : 'btn-blue';

  const statCards = [
    { label: 'Spécialités apprises', value: apiHealth?.mistral_info?.specialties_learned ?? (modelInfo?.training_stats?.class_distribution ? Object.keys(modelInfo.training_stats.class_distribution).length : 0), icon: <Stethoscope size={24} /> },
    { label: 'Modèle de base', value: modelInfo?.base_model ?? 'Inconnu', icon: <Cpu size={24} /> },
    { label: 'Efficacité LoRA', value: apiHealth?.memory_usage?.lora?.lora_efficiency ?? 'N/A', icon: <TrendingUp size={24} /> },
    { label: 'Samples d\'entraînement', value: apiHealth?.mistral_info?.training_samples ?? (modelInfo?.training_stats?.total_samples ?? 0), icon: <Database size={24} /> },
  ];
  
  const statusDisplay = (
    <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold ${getStatusClasses(apiHealth?.status === 'healthy' && apiHealth.model_status === 'loaded' ? 'healthy' : trainingStatus === 'training' ? 'training' : 'error')}`}>
      {trainingStatus === 'training' ? (
        <span className="h-2 w-2 rounded-full bg-white animate-pulse" />
      ) : (
        <span className="h-2 w-2 rounded-full bg-white" />
      )}
      {apiHealth?.status === 'healthy' && apiHealth.model_status === 'loaded' ? 'API et Modèle OK' : trainingStatus === 'training' ? 'Entraînement en cours...' : 'Erreur API ou Modèle non chargé'}
    </div>
  );

  return (
    <div className="medical-classifier-container min-h-screen bg-gray-50 text-gray-900">
      <header className="bg-white shadow-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <Stethoscope className="text-blue-600 h-8 w-8" />
            <h1 className="text-2xl font-bold text-gray-900">
              Classificateur Médical IA
            </h1>
            <div className="hidden md:flex">{statusDisplay}</div>
          </div>
     
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row md:justify-between md:items-center space-y-4 md:space-y-0">
         
          <div className="flex justify-center md:justify-end space-x-2 bg-white rounded-xl shadow-sm p-1">
            <button onClick={() => setActiveTab('predict')} className={getTabClass('predict')}>
              <Send className="h-4 w-4 mr-2 inline-block" />
              Prédiction Unique
            </button>
            <button onClick={() => setActiveTab('batch')} className={getTabClass('batch')}>
              <FileText className="h-4 w-4 mr-2 inline-block" />
              Prédiction par Lot
            </button>
            <button onClick={() => setActiveTab('train')} className={getTabClass('train')}>
              <Brain className="h-4 w-4 mr-2 inline-block" />
              Entraîner le Modèle
            </button>
            <button onClick={() => setActiveTab('stats')} className={getTabClass('stats')}>
              <BarChart3 className="h-4 w-4 mr-2 inline-block" />
              Statistiques
            </button>
          </div>
        </div>
        
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {statCards.map((card, index) => (
            <div key={index} className={`flex items-start p-6 rounded-2xl shadow-xl transition-all duration-300 hover:shadow-2xl ${getStatCardColor(index)}`}>
              <div className={`p-3 rounded-full flex-shrink-0 ${getIconColorClass(card.label)} shadow-blue-500/30`}>
                <div className="text-white h-6 w-6">{card.icon}</div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-slate-700 mb-1 leading-relaxed">{card.label}</p>
                <p className="text-2xl font-bold text-slate-900">{card.value}</p>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-12">
          {/* Section Prédiction Unique */}
          {activeTab === 'predict' && (
            <div className="bg-white p-8 rounded-2xl shadow-xl">
              <h3 className="text-2xl font-semibold mb-6 text-gray-800">
                Prédiction de Spécialité Médicale
              </h3>
              <div className="space-y-6">
                <div>
                  <label htmlFor="text-input" className="block text-sm font-medium text-gray-700 mb-2">
                    Description du cas médical :
                  </label>
                  <textarea
                    id="text-input"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    rows={6}
                    className="w-full border border-gray-300 rounded-lg p-3 text-gray-700 resize-none focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Entrez un cas médical pour prédire la spécialité correspondante..."
                  />
                </div>
                <div className="flex justify-end">
                  <button
                    onClick={handleSinglePrediction}
                    disabled={loading || trainingStatus === 'training' || apiHealth?.model_status !== 'loaded'}
                    className={`flex items-center px-6 py-3 rounded-xl text-white font-semibold shadow-lg transition-all duration-300 hover:shadow-xl ${predictButtonClass}`}
                  >
                    {loading ? (
                      <span className="animate-spin mr-2">
                        <svg className="w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                      </span>
                    ) : (
                      <Zap className="h-5 w-5 mr-2" />
                    )}
                    Prédire la Spécialité
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Section Prédiction par Lot */}
          {activeTab === 'batch' && (
            <div className="bg-white p-8 rounded-2xl shadow-xl">
              <h3 className="text-2xl font-semibold mb-6 text-gray-800">
                Prédiction par Lot
              </h3>
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Importer un fichier texte (.txt) :
                  </label>
                  <label className="flex items-center justify-center w-full px-4 py-6 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors duration-200">
                    <div className="text-center">
                      <Upload className="mx-auto h-10 w-10 text-gray-400" />
                      <p className="mt-2 text-sm text-gray-600">
                        <span className="font-semibold text-blue-600">Cliquez pour téléverser</span> ou glissez-déposez
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        Un cas par ligne.
                      </p>
                    </div>
                    <input type="file" className="hidden" accept=".txt" onChange={handleFileUpload} />
                  </label>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Ou entrez les cas manuellement :
                  </label>
                  <div className="space-y-3 max-h-80 overflow-y-auto pr-2">
                    {batchTexts.map((text, index) => (
                      <div key={index} className="flex items-center space-x-2">
                        <textarea
                          value={text}
                          onChange={(e) => updateBatchInput(index, e.target.value)}
                          rows={2}
                          className="flex-1 border border-gray-300 rounded-lg p-2 text-sm text-gray-700 resize-none focus:ring-blue-500 focus:border-blue-500"
                          placeholder="Entrez un cas médical..."
                        />
                        <button
                          onClick={() => removeBatchInput(index)}
                          className="p-2 text-red-600 hover:text-red-800 transition-colors duration-200"
                        >
                          <Trash2 className="h-5 w-5" />
                        </button>
                      </div>
                    ))}
                  </div>
                  <div className="mt-4">
                    <button
                      onClick={addBatchInput}
                      className="px-4 py-2 text-sm font-semibold text-blue-600 border border-blue-600 rounded-lg hover:bg-blue-50 transition-colors duration-200"
                    >
                      Ajouter un cas
                    </button>
                  </div>
                </div>
                <div className="flex justify-end mt-6">
                  <button
                    onClick={handleBatchPrediction}
                    disabled={loading || trainingStatus === 'training' || apiHealth?.model_status !== 'loaded'}
                    className={`flex items-center px-6 py-3 rounded-xl text-white font-semibold shadow-lg transition-all duration-300 hover:shadow-xl ${predictButtonClass}`}
                  >
                    {loading ? (
                      <span className="animate-spin mr-2">
                        <svg className="w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                      </span>
                    ) : (
                      <Play className="h-5 w-5 mr-2" />
                    )}
                    Lancer la Prédiction par Lot
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Section Entraînement */}
          {activeTab === 'train' && (
            <div className="bg-white p-8 rounded-2xl shadow-xl">
              <h3 className="text-2xl font-semibold mb-6 text-gray-800">
                Entraîner le Modèle Mistral  + LoRA
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div>
                  <label htmlFor="csv-path" className="block text-sm font-medium text-gray-700 mb-1">
                    Chemin du fichier CSV
                  </label>
                  <input
                    type="text"
                    id="csv-path"
                    value={trainingConfig.csv_path}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, csv_path: e.target.value })}
                    className="w-full border border-gray-300 rounded-lg p-2 text-gray-700 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label htmlFor="epochs" className="block text-sm font-medium text-gray-700 mb-1">
                    Nombre d'Époques
                  </label>
                  <input
                    type="number"
                    id="epochs"
                    value={trainingConfig.epochs}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, epochs: parseInt(e.target.value) || 1 })}
                    className="w-full border border-gray-300 rounded-lg p-2 text-gray-700 focus:ring-blue-500 focus:border-blue-500"
                    min="1"
                  />
                </div>
                <div>
                  <label htmlFor="lora-r" className="block text-sm font-medium text-gray-700 mb-1">
                    Paramètre LoRA r
                  </label>
                  <input
                    type="number"
                    id="lora-r"
                    value={trainingConfig.lora_r}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, lora_r: parseInt(e.target.value) || 8 })}
                    className="w-full border border-gray-300 rounded-lg p-2 text-gray-700 focus:ring-blue-500 focus:border-blue-500"
                    min="1"
                  />
                </div>
                <div>
                  <label htmlFor="lora-alpha" className="block text-sm font-medium text-gray-700 mb-1">
                    Paramètre LoRA alpha
                  </label>
                  <input
                    type="number"
                    id="lora-alpha"
                    value={trainingConfig.lora_alpha}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, lora_alpha: parseInt(e.target.value) || 16 })}
                    className="w-full border border-gray-300 rounded-lg p-2 text-gray-700 focus:ring-blue-500 focus:border-blue-500"
                    min="1"
                  />
                </div>
              </div>
              <div className="flex justify-between items-center mt-6">
                <div className="text-sm text-gray-600">
                  <p>
                    <Info className="inline-block h-4 w-4 mr-1 text-blue-500" /> L'entraînement peut prendre plusieurs minutes.
                  </p>
                </div>
                <button
                  onClick={startMistralTraining}
                  disabled={trainingStatus === 'training' || loading}
                  className={`flex items-center px-6 py-3 rounded-xl text-white font-semibold shadow-lg transition-all duration-300 hover:shadow-xl ${trainingButtonClass}`}
                >
                  {trainingStatus === 'training' ? (
                    <span className="animate-spin mr-2">
                      <svg className="w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    </span>
                  ) : (
                    <Brain className="h-5 w-5 mr-2" />
                  )}
                  {trainingStatus === 'training' ? 'Entraînement en cours...' : 'Lancer l\'Entraînement'}
                </button>
              </div>
              <div className="mt-8">
                <h4 className="text-lg font-semibold text-gray-800 mb-3">Logs d'Entraînement</h4>
                <div className="terminal p-4 rounded-xl text-xs overflow-y-auto h-40">
                  {trainingLogs.length > 0 ? (
                    trainingLogs.map((log, index) => (
                      <p key={index} className="mb-1">{`> ${log}`}</p>
                    ))
                  ) : (
                    <p className="text-gray-400">Aucun log disponible.</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Section Statistiques */}
          {activeTab === 'stats' && (
            <div className="bg-white p-8 rounded-2xl shadow-xl">
              <h3 className="text-2xl font-semibold mb-6 text-gray-800">
                Statistiques du Modèle Mistral
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="metric-card p-6 rounded-2xl shadow-lg">
                  <div className="flex items-center mb-4">
                    <Shield className="h-6 w-6 text-blue-600 mr-3" />
                    <h4 className="text-xl font-semibold text-gray-900">État du Service</h4>
                  </div>
                  <div className="space-y-4 text-sm text-gray-700">
                    <p className="flex items-center justify-between">
                      Statut de l'API:
                      <span className={`px-2 py-1 rounded-full text-xs font-bold ${apiHealth?.status === 'healthy' ? 'bg-emerald-100 text-emerald-800' : 'bg-red-100 text-red-800'}`}>
                        {apiHealth?.status ? (apiHealth.status === 'healthy' ? 'Opérationnel' : 'Erreur') : 'Inconnu'}
                      </span>
                    </p>
                    <p className="flex items-center justify-between">
                      Modèle Chargé:
                      <span className={`px-2 py-1 rounded-full text-xs font-bold ${apiHealth?.model_status === 'loaded' ? 'bg-emerald-100 text-emerald-800' : 'bg-red-100 text-red-800'}`}>
                        {apiHealth?.model_status === 'loaded' ? 'Oui' : 'Non'}
                      </span>
                    </p>
                    <p className="flex items-center justify-between">
                      Accélération GPU:
                      <span className={`px-2 py-1 rounded-full text-xs font-bold ${apiHealth?.cuda_available ? 'bg-emerald-100 text-emerald-800' : 'bg-red-100 text-red-800'}`}>
                        {apiHealth?.cuda_available ? 'Activée' : 'Non'}
                      </span>
                    </p>
                  </div>
                </div>

                <div className="metric-card p-6 rounded-2xl shadow-lg">
                  <div className="flex items-center mb-4">
                    <Settings className="h-6 w-6 text-orange-600 mr-3" />
                    <h4 className="text-xl font-semibold text-gray-900">Configuration LoRA</h4>
                  </div>
                  <div className="space-y-4 text-sm text-gray-700">
                    <p className="flex items-center justify-between">
                      Rang (r):
                      <span className="font-mono text-gray-900">{modelInfo?.lora_config?.r ?? 'N/A'}</span>
                    </p>
                    <p className="flex items-center justify-between">
                      Alpha:
                      <span className="font-mono text-gray-900">{modelInfo?.lora_config?.alpha ?? 'N/A'}</span>
                    </p>
                    <p className="flex items-center justify-between">
                      Modules ciblés:
                      <span className="font-mono text-gray-900">{modelInfo?.lora_config?.target_modules.length ?? 'N/A'}</span>
                    </p>
                    <p className="flex items-center justify-between">
                      Paramètres Entraînables:
                      <span className="font-mono text-gray-900">{(modelInfo?.lora_config?.trainable_params ?? 0).toLocaleString()}</span>
                    </p>
                    <p className="flex items-center justify-between">
                      Taille du Modèle Original:
                      <span className="font-mono text-gray-900">{(modelInfo?.lora_config?.total_params ?? 0).toLocaleString()}</span>
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-6 metric-card p-6 rounded-2xl shadow-lg">
                <div className="flex items-center mb-4">
                  <BarChart3 className="h-6 w-6 text-purple-600 mr-3" />
                  <h4 className="text-xl font-semibold text-gray-900">Distribution des Spécialités</h4>
                </div>
                <div className="space-y-4">
                  {specialties.length > 0 ? (
                    specialties.map((specialty, index) => (
                      <div key={specialty.class_id} className="flex items-center">
                        <div className={`w-3 h-3 rounded-full mr-3 ${getDotColorClass(specialty.specialty_name)}`} />
                        <span className="flex-1 text-sm font-medium text-gray-700 truncate">{specialty.specialty_name}</span>
                        <div className="w-16 h-1.5 rounded-full bg-gray-200">
                          <div
                            className={`h-full rounded-full ${getProgressBarColor(index)}`}
                            style={{ width: `${(specialty.training_samples / (modelInfo?.training_stats?.total_samples ?? 1)) * 100}%` }}
                          />
                        </div>
                        <span className="ml-4 text-xs font-mono text-gray-600 w-12 text-right">{specialty.training_samples}</span>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-gray-500">Aucune spécialité trouvée. Veuillez d'abord lancer l'entraînement.</p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
        
        {/* Section Résultats de Prédiction */}
        {predictions.length > 0 && (
          <div className="mt-12">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-2xl font-semibold text-gray-800">Résultats de Prédiction</h3>
              <div className="flex space-x-2">
                <button
                  onClick={handleDownload}
                  className="px-4 py-2 text-sm font-semibold text-gray-700 bg-gray-200 rounded-lg hover:bg-gray-300 transition-colors duration-200 flex items-center"
                >
                  <Download className="h-4 w-4 mr-2" />
                  Télécharger JSON
                </button>
                <button
                  onClick={handlePredictionClear}
                  className="px-4 py-2 text-sm font-semibold text-red-600 bg-red-100 rounded-lg hover:bg-red-200 transition-colors duration-200 flex items-center"
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Effacer
                </button>
              </div>
            </div>
            
            <div className="space-y-6">
              {predictions.map((pred, index) => (
                <div key={index} className={`specialty-card p-6 rounded-2xl shadow-lg transition-transform duration-300 ${getPredictionCardColor(pred.confidence)}`}>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <Stethoscope className="h-6 w-6 text-gray-800" />
                      <p className="text-xl font-bold text-gray-900">{pred.predicted_specialty}</p>
                    </div>
                    <div className="flex items-center space-x-2">
                      <p className={`font-semibold text-lg ${getConfidenceColor(pred.confidence)}`}>
                        {`${(pred.confidence * 100).toFixed(2)}%`}
                      </p>
                      <span className={`px-3 py-1 rounded-full text-xs font-semibold text-white ${getConfidenceBadgeClass(pred.confidence)}`}>
                        {pred.confidence > 0.8 ? 'Élevé' : pred.confidence > 0.5 ? 'Moyen' : 'Faible'}
                      </span>
                    </div>
                  </div>
                  
                  <div className="border-t border-gray-200 pt-4 mt-4">
                    <p className="text-sm font-medium text-gray-700 mb-2">Cas soumis:</p>
                    <blockquote className="border-l-4 border-gray-300 pl-4 text-gray-600 italic">
                      <p className="line-clamp-4 leading-relaxed">{pred.text}</p>
                    </blockquote>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 text-sm text-gray-500">
                    <div className="flex items-center space-x-2">
                      <Clock className="h-4 w-4" />
                      <span>{pred.processing_time.toFixed(2)}s</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4" />
                      <span>{new Date(pred.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Cpu className="h-4 w-4" />
                      <span>{pred.model_type}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Info className="h-4 w-4" />
                      <span>{pred.mistral_metadata?.base_model}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <footer className="bg-gray-100 mt-12 py-8 text-sm text-gray-600 text-center">
        <p>Développé avec ❤️ et Mistral</p>
        <p className="mt-2">© 2025</p>
      </footer>
    </div>
  );
};

export default MistralMedicalClassifierApp;
