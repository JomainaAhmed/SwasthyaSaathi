# 🏥 स्वास्थ्य साथी | SwasthyaSaathi (Health Companion)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-green.svg)](https://xgboost.readthedocs.io/)
[![FAISS](https://img.shields.io/badge/FAISS-1.7.4-orange.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AI-Powered Rural Healthcare Chatbot with Hybrid ML-RAG Architecture**

A bilingual (Hindi / English), privacy-first health assistant specifically designed for low-connectivity rural environments in India. It combines fast **XGBoost** symptom classification with **FAISS-powered Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware, and educational health guidance with emergency escalation.

---

## ✨ Key Features

### 🧠 Hybrid ML-RAG Symptom Analysis
* **Lightweight Machine Learning**: Instant inference (<30ms) powered by TF-IDF + XGBoost trained across 24 disease classes.
* **Semantic RAG Fallback**: Uses FAISS (`IndexFlatIP` cosine similarity) and `Sentence Transformers` (`paraphrase-MiniLM-L3-v2`) to retrieve detailed medical guidance from a verified knowledge base when confidence is low.
* **Confidence Routing**: Intelligent switching threshold ($\ge 0.60$) ensuring reliable advice.

### 🌍 Environmental & Contextual Intelligence
* **Weather & AQI Integration**: Live OpenWeatherMap data to evaluate temperature, humidity, and air quality risk factors.
* **Low-Bandwidth Optimization**: 3-hour weather and 2-hour AQI local caching to minimize cellular data consumption.
* **Offline Fallbacks**: Graceful degradation to representative Indian regional cities and monthly seasonal averages (`SEASONAL_AVERAGES`) when offline.

### 🗣️ Multilingual & Inclusive
* **Bilingual UI**: Seamless real-time switching between English and Hindi (हिंदी).
* **Voice Input**: Web Speech API integration embedded via Streamlit components for users with low literacy.
* **Symptom Severity Slider**: Interactive 1–10 severity slider with tailored empathetic responses.
* **PDF Health Summaries**: One-click downloadable patient summary report to share with healthcare workers.

### 🚨 Clinical Safety & Triage
* **Rule-based Triage Classifier**: Automatically flags red-flag emergencies (e.g. chest pain, difficulty breathing, high fever $>103^\circ\text{F}$, prolonged illness $>7$ days).
* **Emergency Assistance**: One-click emergency guidance and hospital referral prompts.
* **100% Private**: In-memory session state; no patient data is permanently stored on disk or external servers.

---

## 🏗️ Architecture

```
User Input (Text / Voice in Hindi or English)
       │
       ├──► Severity Classifier (Emergency checks: Triage / Hospital alerts)
       │
       ▼
Hybrid Chatbot Router
       │
   [Confidence >= 0.60?]
       ├──► YES ──► XGBoost Text Predictor ──► Match Knowledge Base Entity
       └──► NO  ──► FAISS Vector Search ──► Paraphrase-MiniLM Semantic Retrieval
                           │
                           ▼
                  Weather & AQI Risk Context
                           │
                           ▼
             Empathetic Response + Clinical Triage + PDF Export
```

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend** | Streamlit, Custom Healthcare CSS, Streamlit Components (Web Speech API) |
| **Machine Learning** | XGBoost, scikit-learn (SVM, Logistic Regression, Naive Bayes, Random Forest) |
| **Vector Search (RAG)**| FAISS (CPU), Sentence-Transformers (`paraphrase-MiniLM-L3-v2`) |
| **APIs & Data** | OpenWeatherMap API, Pandas, NumPy |
| **Reporting** | FPDF2 |

---

## 📁 Project Structure

```text
SwasthyaSaathi/
├── app.py                       # Main Streamlit web application
├── config.py                    # Environment settings, cache durations & offline averages
├── hybrid_chatbot_lightweight.py# Decision router (ML vs FAISS RAG fallback)
├── lightweight_predictor.py     # XGBoost + TF-IDF symptom classifier
├── lightweight_rag.py           # FAISS index builder & semantic search engine
├── knowledge_base.py / .json    # 24-disease verified medical encyclopedia
├── severity_classifier.py       # Triage scoring & red-flag symptom detection
├── weather_module.py            # Weather & AQI module with local cache & seasonal fallback
├── ui_components.py             # UI helpers, Hindi/English translations, voice & PDF export
├── Symptom2Disease.csv          # Training dataset for symptom classification
├── requirements.txt             # Project dependencies
├── .gitignore                   # Ignored files (caches, logs, virtual environments)
│
├── models/                      # Pre-trained models (included & ready to run)
│   ├── xgb_text_model.pkl       # Serialized XGBoost model (~2.3 MB)
│   ├── tfidf_vectorizer.pkl     # TF-IDF feature extractor
│   ├── label_encoder.pkl        # Label encoder for 24 diseases
│   ├── faiss_index.bin          # Pre-built FAISS vector index
│   ├── rag_metadata.pkl         # RAG metadata mapping
│   └── trained/                 # Benchmark models from comparative study
│
├── evaluation/                  # Research paper benchmark metrics & figures
│   ├── figures/                 # ROC curves, confusion matrices, radar charts
│   ├── model_comparison.csv     # 8-model performance benchmark metrics
│   ├── model_comparison.json    # JSON export of evaluation metrics
│   └── latex_table.txt          # Ready-to-publish LaTeX comparison table
│
├── train_text_model.py          # Script to retrain primary XGBoost model
├── train_all_models.py          # Multi-model comparative training suite
├── evaluate_all_models.py       # Evaluation & cross-validation benchmark script
└── generate_paper_graphs.py     # Figure generation for academic publication
```

---

## 📊 Research & Model Performance

Comparative analysis across 8 machine learning algorithms on the `Symptom2Disease` dataset:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM** | **95.4%** | 96.0% | 95.4% | 95.5% | 0.999 | ~30 ms |
| **Logistic Regression** | 93.3% | 94.0% | 93.3% | 93.3% | 0.998 | ~1 ms |
| **Multinomial Naive Bayes** | 91.2% | 92.4% | 91.2% | 90.7% | 0.997 | < 1 ms |
| **LightGBM** | 87.5% | 88.4% | 87.5% | 87.4% | 0.994 | ~19 ms |
| **XGBoost (Production)** | 87.5% | 87.8% | 87.5% | 87.3% | 0.993 | ~19 ms |
| **Neural Network (MLP)** | 86.2% | 88.4% | 86.2% | 85.4% | 0.993 | ~8 ms |
| **Random Forest** | 85.8% | 86.7% | 85.8% | 85.6% | 0.993 | ~61 ms |
| **CatBoost** | 85.0% | 85.4% | 85.0% | 84.2% | 0.990 | ~6 ms |

* **Hybrid System Accuracy**: **~92%** (with FAISS semantic fallback)
* **Model Footprint**: ~2.3 MB (easily runs on low-spec edge hardware or smartphones)

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/JomainaAhmed/SwasthyaSaathi.git
cd SwasthyaSaathi
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure Environment Variables
Set your OpenWeatherMap API key:
```bash
# Windows PowerShell
$env:OPENWEATHER_API_KEY="your_api_key_here"

# Linux / macOS
export OPENWEATHER_API_KEY="your_api_key_here"
```
*(If no API key is provided, the application will automatically use built-in seasonal averages and cached representative data).*

### 5. Launch the Application
Pre-trained models and FAISS indexes are already included in `models/`. Launch the Streamlit app directly:
```bash
streamlit run app.py
```

---

## 🔄 Retraining & Reproducibility (Optional)

To retrain models from scratch or regenerate the research figures:

1. **Retrain the production text classifier**:
   ```bash
   python train_text_model.py
   ```
2. **Rebuild the FAISS vector index**:
   ```bash
   python lightweight_rag.py
   ```
3. **Train and evaluate all 8 research models**:
   ```bash
   python train_all_models.py
   python evaluate_all_models.py
   python generate_paper_graphs.py
   ```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Author & Contact

* **Author**: [Jomaina Ahmed](https://github.com/JomainaAhmed)
* **GitHub**: [@JomainaAhmed](https://github.com/JomainaAhmed)
* **Email**: [jomaina.ahmed@gmail.com](mailto:jomaina.ahmed@gmail.com)

⭐ *If you found this project helpful, please consider giving it a star on GitHub!*
