# Samsung EnnovateX 2025 AI Challenge Submission

- **Problem Statement** - *Real-time Detection of Reel Traffic vs Non-reel Traffic in a Social-networking Application*
- **Team name** - *Synapse Squad*
- **Team members (Names)** - *NALAM VENKATA SURYA HARSHITH*, *MATTA PRIYUSHA*, *BONDADA MADHULIKA SREE*, *KOTHAPALLI SNEHITH REDDY* 
- **Demo Video Link** - *(https://www.youtube.com/watch?v=OM2ceuoORro)*


### Project Artefacts

### Project Artefacts  

- **Technical Documentation** - [Docs](docs)  
*(All technical details, design choices, and architecture are documented inside the `docs/` folder in markdown format.  
The documentation includes the following:  
  - Problem statement and motivation  
  - System design and architecture diagrams  
  - Data preprocessing pipeline explanation  
  - Feature engineering details  
  - Model selection, training process, and evaluation metrics  
  - Instructions for running the project, environment setup, and dependencies  
  - Limitations, ethical considerations, and scalability analysis  
This ensures that anyone reviewing or replicating the project can fully understand both the implementation and rationale.)*  

---

- **Source Code** - [Source](src)  
*(All source code is maintained inside the `src/` folder.  
The code is modular and organized into subcomponents:  
  - **data_preprocessing/** → scripts for cleaning and preparing datasets  
  - **models/** → training scripts, saved model checkpoints, evaluation scripts  
  - **ui/** → Streamlit app code for user interaction  
  - **utils/** → helper functions, configuration files, logging, etc.  
The codebase is written in Python and can be successfully installed/executed on intended platforms. A `requirements.txt` file and installation guide are provided for reproducibility. The code has been tested for consistent execution and real-time predictions.)*  

---

- **Models Used**  
*(The project leverages multiple machine learning models to classify streaming vs non-streaming traffic in real-time. The models include:  
  - Random Forest Classifier (Scikit-learn) – chosen for its balance between accuracy and efficiency  
  - Logistic Regression – used as a lightweight baseline model  
  - XGBoost Classifier – employed for gradient boosting performance and feature importance analysis  
We have relied on open-weight models available in the ML ecosystem, and Hugging Face links to pre-trained versions (if applicable) will be added here.)*  

---

- **Models Published**  
*(If new models are developed or fine-tuned as part of this solution, they will be uploaded to Hugging Face under an appropriate open-source license (e.g., Apache 2.0, MIT, or Creative Commons).  
The published models will include metadata such as:  
  - Description of the model  
  - Intended use cases  
  - Training datasets and methodology  
  - Performance benchmarks  
Links will be added here once uploaded.)*  

---

- **Datasets Used**  
*(The following publicly available datasets were used for training and evaluation:  
  - [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html) – contains labeled internet traffic flows including streaming and non-streaming activities  
  - [ISCX VPN-nonVPN Dataset](https://www.unb.ca/cic/datasets/vpn.html) – provides traffic patterns from VPN and non-VPN sources to test generalization  
Both datasets are publicly available under open data licenses (Creative Commons/Open Data Commons), making them suitable for academic and research use.)*  

---

- **Datasets Published**  
*(If synthetic datasets or processed versions of raw traffic flows are generated during the project, they will be published on Hugging Face with the following details:  
  - Dataset description and purpose  
  - Schema and feature explanation (e.g., flow duration, packets/sec, inter-arrival times)  
  - License under which it is shared (Creative Commons / Open Data Commons)  
  - Example usage code snippets  
This ensures transparency and reproducibility for future researchers and developers.)*  

---

### Attribution  

This project builds on top of several open-source tools and frameworks that made development possible:  
- [Streamlit](https://streamlit.io/) – used to build the interactive user interface for live testing and demonstrations  
- [Scikit-learn](https://scikit-learn.org/stable/) – for building and evaluating traditional ML models (Random Forest, Logistic Regression)  
- [XGBoost](https://xgboost.readthedocs.io/) – for high-performance gradient boosting classification  
- [Plotly](https://plotly.com/python/) – for generating interactive visualizations of traffic features and model outputs  

**New Features Developed by Our Team:**  
- Designed and implemented a **real-time traffic classification pipeline** capable of distinguishing streaming vs non-streaming flows.  
- Built an **interactive Streamlit UI** where users can test models by inputting sample traffic features.  
- Developed **visual confidence gauges and feature importance graphs** for explainability.  
- Created a **lightweight Random Forest model** with optimized features suitable for deployment on edge devices and routers.  
- Addressed **ethical concerns** by avoiding deep packet inspection and ensuring privacy-preserving classification.  

