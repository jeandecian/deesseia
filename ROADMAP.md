# Roadmap

## Released Versions

| Version    | Phase       | Main Focus                    | Key Features                                                                                                                                           | Deferred Features                    |
| ---------- | ----------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| **v1.0.0** | Foundations | Core Foundation               | Data loading (CSV, JSON, Parquet, SQL), data inspection, cleaning, validation, and fake data generator                                                 | —                                    |
| **v1.1.0** | Foundations | Feature Scaling & Encoding    | Feature scaling (MinMax, Standard, Robust), and encoding (label, one-hot, target, frequency)                                                           | Excel loading (v1.0.0)               |
| **v1.2.0** | Foundations | Imputation & Feature Creation | Imputation (mean, median, KNN, model-based), polynomial features, and interaction features                                                             | —                                    |
| **v1.3.0** | Foundations | Cross-Validation              | K-Fold, Stratified K-Fold, and Time Series split                                                                                                       | —                                    |
| **v1.4.0** | Foundations | EDA & Descriptive Statistics  | Descriptive statistics (mean, median, mode, std, quartiles, IQR), distribution analysis (histograms, KDE, boxplots), and automated insights generation | Train/validation/test split (v1.3.0) |

## v1.x.x - Foundations

Building the essential foundation for any data science project.

### Version 1.5.0 — Correlation & Hypothesis Testing

**Phase**: Foundations

**Focus**: Measuring relationships and statistical testing

**Status**: In Development

**Key Features**:

- Correlation matrix and heatmaps
- Hypothesis testing (t-test, ANOVA, chi-square, Mann-Whitney)
- Normality tests (Shapiro-Wilk, Kolmogorov-Smirnov)
- Missing value pattern analysis

---

### Version 1.6.0 — Visualization & Geospatial

**Phase**: Foundations

**Focus**: Creating visualizations and working with geospatial data

**Status**: Planned

**Key Features**:

- Statistical plots (Matplotlib, Seaborn)
- Interactive plots (Plotly)
- Geospatial analysis (GeoPandas)
- Interactive dashboards
- Choropleth maps
- Hexbin maps
- Sankey diagrams

## v2.x.x — Core Machine Learning

All essential supervised and unsupervised learning algorithms.

### Version 2.0.0 — ML Fundamentals

**Phase**: Core ML

**Focus**: Foundational concepts for machine learning

**Status**: Planned

**Key Features**:

- Bias-variance tradeoff analysis
- Regularization (L1, L2, Elastic Net)
- Hyperparameter tuning (Grid Search, Random Search, Bayesian)
- Learning curves
- Model persistence (pickle, joblib, ONNX)

---

### Version 2.1.0 — Linear Regression

**Phase**: Core ML

**Focus**: Linear and regularized regression models

**Status**: Planned

**Key Features**:

- Linear Regression
- Ridge Regression (L2)
- Lasso Regression (L1)
- ElasticNet

---

### Version 2.2.0 — Tree-based Regression

**Phase**: Core ML

**Focus**: Decision tree and ensemble regression models

**Status**: Planned

**Key Features**:

- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

---

### Version 2.3.0 — Advanced Regression

**Phase**: Core ML

**Focus**: Non-linear and specialized regression models

**Status**: Planned

**Key Features**:

- Polynomial Regression
- Bayesian Linear Regression
- Support Vector Regression (SVR)

---

### Version 2.4.0 — Linear Classification

**Phase**: Core ML

**Focus**: Linear classifiers for binary and multiclass problems

**Status**: Planned

**Key Features**:

- Logistic Regression (binary, multinomial)
- Naive Bayes (Gaussian, Multinomial, Bernoulli)
- Linear SVM
- K-Nearest Neighbors (KNN)

---

### Version 2.5.0 — Tree-based Classification

**Phase**: Core ML

**Focus**: Decision tree and ensemble classification models

**Status**: Planned

**Key Features**:

- Decision Trees
- Random Forest
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier

---

### Version 2.6.0 — Advanced Classification

**Phase**: Core ML

**Focus**: Kernel methods and handling imbalanced data

**Status**: Planned

**Key Features**:

- Support Vector Machines (SVM) with kernels
- Class imbalance handling (SMOTE, ADASYN, class_weight)

---

### Version 2.7.0 — Clustering

**Phase**: Core ML

**Focus**: Unsupervised clustering algorithms

**Status**: Planned

**Key Features**:

- K-Means Clustering (with elbow method)
- Hierarchical Clustering (with dendrograms)
- DBSCAN Clustering
- Gaussian Mixture Models (GMM)

---

### Version 2.8.0 — Dimensionality Reduction

**Phase**: Core ML

**Focus**: Reducing feature space while preserving information

**Status**: Planned

**Key Features**:

- Principal Component Analysis (PCA)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)
- Linear Discriminant Analysis (LDA)

---

### Version 2.9.0 — Model Evaluation & MLOps

**Phase**: Core ML

**Focus**: Comprehensive evaluation, explainability, monitoring

**Status**: Planned

**Key Features**:

- All classification metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
- All regression metrics (MAE, MSE, RMSE, R², MAPE, SMAPE, MASE)
- Ranking metrics (nDCG, MRR, Recall@k)
- Clustering evaluation (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Model calibration (Reliability diagrams, Platt scaling, Temperature scaling)
- Explainability (SHAP, LIME, Feature Importance)
- Drift detection (Data drift, Concept drift)
- Model monitoring
- Experiment tracking (MLflow integration)

## v3.x.x — Advanced Machine Learning

Deep learning, NLP, LLMs, computer vision, and audio processing.

### Version 3.0.0 — Neural Network Fundamentals

**Phase**: Advanced ML

**Focus**: Core neural network building blocks

**Status**: Planned

**Key Features**:

- Neural networks (MLP, layers, activations)
- Loss functions
- Backpropagation
- Optimizers

---

### Version 3.1.0 — Convolutional Neural Networks

**Phase**: Advanced ML

**Focus**: CNNs for image and spatial data

**Status**: Planned

**Key Features**:

- Convolutional Neural Networks (CNNs)
- ResNet
- VGG
- EfficientNet

---

### Version 3.2.0 — Sequential Models

**Phase**: Advanced ML

**Focus**: RNNs and their variants for sequence data

**Status**: Planned

**Key Features**:

- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)
- Bidirectional RNNs

---

### Version 3.3.0 — Transformers & Attention

**Phase**: Advanced ML

**Focus**: Attention mechanisms and transformer architecture

**Status**: Planned

**Key Features**:

- Attention mechanisms
- Multi-head attention
- Transformer architecture

---

### Version 3.4.0 — Generative Models

**Phase**: Advanced ML

**Focus**: Generating new data from learned distributions

**Status**: Planned

**Key Features**:

- Autoencoders
- Variational Autoencoders (VAEs)
- Generative Adversarial Networks (GANs)

---

### Version 3.5.0 — Transfer Learning

**Phase**: Advanced ML

**Focus**: Leveraging pre-trained models

**Status**: Planned

**Key Features**:

- Transfer learning (fine-tuning, feature extraction)

---

### Version 3.6.0 — NLP Fundamentals

**Phase**: Advanced ML

**Focus**: Text preprocessing for NLP tasks

**Status**: Planned

**Key Features**:

- Text preprocessing (tokenization, stemming, lemmatization)

---

### Version 3.7.0 — Text Representations

**Phase**: Advanced ML

**Focus**: Converting text to numerical representations

**Status**: Planned

**Key Features**:

- Text representations (Bag-of-Words, TF-IDF)
- Word embeddings (Word2Vec, GloVe, FastText)
- Sentence embeddings (Sentence Transformers)

---

### Version 3.8.0 — Core NLP Tasks

**Phase**: Advanced ML

**Focus**: Fundamental NLP tasks

**Status**: Planned

**Key Features**:

- Named Entity Recognition (NER)
- Part-of-Speech (POS) tagging
- Dependency parsing

---

### Version 3.9.0 — NLP Applications

**Phase**: Advanced ML

**Focus**: Higher-level NLP applications

**Status**: Planned

**Key Features**:

- Text classification
- Sentiment analysis
- Topic modeling (LDA, BERTopic)
- Text summarization

---

### Version 3.10.0 — Search & Retrieval

**Phase**: Advanced ML

**Focus**: Information retrieval and search systems

**Status**: Planned

**Key Features**:

- Sparse retrieval (BM25, TF-IDF)
- Dense retrieval (embeddings, vector search)
- Hybrid search (RRF, ranking fusion)
- FAISS integration
- Vector databases

---

### Version 3.11.0 — LLMs & Prompt Engineering

**Phase**: Advanced ML

**Focus**: Large Language Models and prompting techniques

**Status**: Planned

**Key Features**:

- Language Models (GPT, LLaMA, Mistral, BERT)
- Prompt engineering (zero-shot, few-shot, Chain-of-Thought)

---

### Version 3.12.0 — Fine-tuning & RAG

**Phase**: Advanced ML

**Focus**: Adapting LLMs to specific tasks

**Status**: Planned

**Key Features**:

- Fine-tuning (SFT, RLHF, PEFT, LoRA, QLoRA)
- Retrieval-Augmented Generation (RAG)
- GraphRAG

---

### Version 3.13.0 — AI Agents

**Phase**: Advanced ML

**Focus**: Autonomous AI agents and tool use

**Status**: Planned

**Key Features**:

- AI agents (tool calling, ReAct, multi-agent)
- LangChain and LangGraph integration
- Evaluation (perplexity, BLEU, ROUGE)

---

### Version 3.14.0 — Computer Vision

**Phase**: Advanced ML

**Focus**: Image classification and object detection

**Status**: Planned

**Key Features**:

- Image classification
- Object detection (YOLO, R-CNN, SSD)

---

### Version 3.15.0 — Segmentation & OCR

**Phase**: Advanced ML

**Focus**: Image segmentation and text extraction

**Status**: Planned

**Key Features**:

- Semantic segmentation
- Instance segmentation
- Panoptic segmentation
- OCR (Tesseract, PaddleOCR)

---

### Version 3.16.0 — Audio Processing

**Phase**: Advanced ML

**Focus**: Speech and audio analysis

**Status**: Planned

**Key Features**:

- Face detection and recognition
- Speech-to-text (STT)
- Text-to-speech (TTS)
- Audio features (MFCC, spectrograms)

## v4.x.x — Specialized Domains

Time series, graph ML, reinforcement learning, Bayesian methods, and optimization.

### Version 4.0.0 — Time Series Fundamentals

**Phase**: Specialized

**Focus**: Core time series analysis

**Status**: Planned

**Key Features**:

- Time series decomposition (trend, seasonality)
- ARIMA
- SARIMA

---

### Version 4.1.0 — Advanced Forecasting

**Phase**: Specialized

**Focus**: Modern forecasting methods

**Status**: Planned

**Key Features**:

- Prophet
- Exponential Smoothing
- LSTM for time series

---

### Version 4.2.0 — Time Series Applications

**Phase**: Specialized

**Focus**: Advanced time series tasks

**Status**: Planned

**Key Features**:

- Anomaly detection
- Forecast reconciliation
- Lag and rolling features

---

### Version 4.3.0 — Graph ML

**Phase**: Specialized

**Focus**: Graph neural networks and knowledge graphs

**Status**: Planned

**Key Features**:

- Graph Neural Networks (GNN)
- Graph Convolutional Networks (GCN)
- Knowledge graphs
- GraphRAG

---

### Version 4.4.0 — Graph Analytics

**Phase**: Specialized

**Focus**: Graph analysis and visualization

**Status**: Planned

**Key Features**:

- Centrality measures (degree, betweenness, closeness)
- Community detection
- Graph visualization

---

### Version 4.5.0 — Reinforcement Learning Fundamentals

**Phase**: Specialized

**Focus**: Core RL concepts and value-based methods

**Status**: Planned

**Key Features**:

- Markov Decision Processes (MDP)
- Q-Learning
- Deep Q-Networks (DQN)

---

### Version 4.6.0 — Advanced RL

**Phase**: Specialized

**Focus**: Policy-based and advanced RL methods

**Status**: Planned

**Key Features**:

- Policy gradients
- Proximal Policy Optimization (PPO)
- Asynchronous Advantage Actor-Critic (A2C)
- Deep Deterministic Policy Gradient (DDPG)
- Experience replay
- Multi-agent RL

---

### Version 4.7.0 — Bayesian Methods

**Phase**: Specialized

**Focus**: Bayesian inference and sampling

**Status**: Planned

**Key Features**:

- Bayesian inference
- Markov Chain Monte Carlo (MCMC)
- Hamiltonian Monte Carlo (HMC)

---

### Version 4.8.0 — Gaussian Processes & Optimization

**Phase**: Specialized

**Focus**: Probabilistic models and optimization algorithms

**Status**: Planned

**Key Features**:

- Gaussian Processes
- Evolutionary algorithms (GA, PSO)
- Multi-criteria optimization
- AutoML

## v5.x.x+ — Continuous Evolution

Long-term maintenance, community-driven development, and emerging technologies.

### Version 5.0.0+ — Continuous Improvement

**Phase**: Evolution

**Focus**: Community-driven development, new algorithms, enhancements

**Status**: Future

**Key Features**:

- Community contributions
- New algorithms and models
- Performance enhancements
- Bug fixes
- Backward-compatible feature additions
- Emerging technology integration
- API refinements based on user feedback
