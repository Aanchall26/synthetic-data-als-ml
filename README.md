🧬 SYNTHETIC DATA AUGMENTATION FOR ALS CLASSIFICATION USING MACHINE LEARNING


📌 PROJECT OVERVIEW
Amyotrophic Lateral Sclerosis (ALS) is a rare neurodegenerative disease with limited publicly available biomedical data. This makes it difficult to train robust machine learning models due to data scarcity and class imbalance.
This project investigates whether synthetic data generation techniques can improve classification performance for ALS detection using gene expression data.
We compare traditional and generative approaches and evaluate their impact on model performance using real-world test data.


🎯 OBJECTIVES
Handle class imbalance in ALS gene expression dataset
Generate synthetic data using multiple approaches
Evaluate ML performance across different data distributions
Compare effectiveness of:
Baseline model (original data)
SMOTE (oversampling)
CTGAN (GAN-based tabular synthesis)
TVAE (Variational Autoencoder)
CopulaGAN (Copula-based GAN)


📊 DATASET INFORMATION
This project uses publicly available gene expression datasets from the NCBI GEO database.
🧬 DATASET SOURCES
ALS Dataset: GSE112681 (NCBI GEO)
            (ALS patient gene expression profiles)
Control Dataset: GSE37171 (NCBI GEO)
            (Healthy control gene expression profiles)
            
📁 RAW DATA FILES
data/raw/als_series_matrix.txt
data/raw/control_series_matrix.txt


⚙️ PREPROCESSING STEPS
Merging ALS and control datasets
Handling missing values
Feature cleaning and normalization
Feature reduction (high-dimensional gene filtering)
Train-test splitting
Label encoding:
ALS = 1
Control = 0


🧠 METHODOLOGY
📊 1. Baseline Model
Trained on original imbalanced dataset
Random Forest Classifier used for classification
⚖️ 2. SMOTE (Synthetic Minority Oversampling Technique)
Generates synthetic samples using nearest neighbors
Balances class distribution in training data
🤖 3. CTGAN (Conditional Tabular GAN)
Deep generative model for tabular data
Learns complex feature distributions
🧬 4. TVAE (Tabular Variational Autoencoder)
Probabilistic deep learning model
Captures latent structure of gene expression data
🔗 5. CopulaGAN
Combines statistical copulas with GAN architecture
Preserves feature dependency structure



🧠 Model Configuration
This project evaluates multiple synthetic data generation techniques with carefully selected hyperparameters to ensure fair comparison.
⚙️ Hyperparameter Settings
| Model   | Key Hyperparameters                                                   |
|---------------------------------------------------------------------------------|
|SMOTE    | k_neighbors = 5–10 (tuned)                                            |
|TGAN     | epochs = 300–500 , batch_size = 256                                   |
|TVAE     | latent_dim = 10–20 , KL annealing enabled                             |
|CopulaGAN| epochs = 300–500 , batch_size = 256 , copula-based dependency modeling|


📁 PROJECT STRUCTURE
ALS/
│
├── src/
│   ├── baseline_evaluation.py
│   ├── smote_evaluation.py
│   ├── ctgan_evaluation.py
│   ├── tvae_evaluation.py
│   ├── copulagan_evaluation.py
│   ├── evaluate_all_models.py
│   ├── train_copulagan.py
│   ├── preprocess_gene_expression.py
│   ├── reduce_features.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── models/
│   ├── copulagan.pkl
│   ├── ctgan.pkl
│   └── tvae.pkl
│
├── results/
│   ├── metrics outputs
│   └── plots
│
├── notebooks/
├── evaluation/
├── README.md
└── requirements.txt

🔁 WORKFLOW PIPELINE
Raw GEO Gene Expression Data
        ↓
Data Cleaning & Preprocessing
        ↓
Feature Reduction
        ↓
Train-Test Split
        ↓
Synthetic Data Generation
(SMOTE / CTGAN / TVAE / CopulaGAN)
        ↓
Random Forest Training
        ↓
Evaluation on Real Test Set
        ↓
Performance Comparison

📊 EVALUATION METRICS
Each model is evaluated using:
Accuracy
Precision
Recall
F1-score
Classification Report
5-Fold Stratified Cross Validation (F1-score)

📌 NOTES
Large datasets and model files are excluded from GitHub using .gitignore
Only code and processed results are tracked in this repository
Synthetic datasets are generated dynamically during execution

📈 FUTURE WORK
Hyperparameter tuning of GAN models
Multi-disease synthetic data extension
Deep learning classifiers
Deployment as a web-based biomedical ML tool


👩‍💻 Author
Aanchal Yadav
B.Tech CSE
Interested in Machine Learning, AI Systems and Healthcare AI


