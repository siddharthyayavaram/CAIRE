<div align="center">

# *CAIRE: Cultural Attribution of Images by Retrieval-Augmented Evaluation*

## Overview  

<p align="center">
  <img src="assets/fig2_caire.jpg" width="100%">
</p>  

</div>

## Abstract

As text-to-image models become increasingly prevalent, ensuring their equitable performance across diverse cultural contexts is critical. 
Efforts to mitigate cross-cultural biases have been hampered by trade-offs, including a loss in performance, factual inaccuracies, or offensive outputs.
Despite widespread recognition of these challenges, an inability to reliably measure these biases has stalled progress. To address this gap, we introduce CAIRE, a novel evaluation metric that assesses the degree of cultural relevance of an image, given a user-defined set of labels. Our framework grounds entities and concepts in the image to a knowledge-base and uses factual information to give independent graded judgments for each culture label.
On a manually curated dataset of culturally salient but rare items built using language models, CAIRE surpasses all baselines by **28%** F1 points. Additionally, we construct two datasets for culturally universal concepts, one comprising of T2I generated outputs and another retrieved from naturally-occurring data. CAIRE achieves Pearson’s correlations of **0.56** and **0.66** with human ratings on these sets, based on a 5-point Likert scale of cultural relevance. This demonstrates its strong alignment with human judgment across diverse image sources.

---

## Installation and Setup

**Prerequisites:**  
- Python version 3.9 or later.

#### Step 1: Clone the Repository

```sh
git clone https://github.com/siddharthyayavaram/CAIRE.git
cd CAIRE
```

#### Step 2: Setup  

The setup process performs the following tasks:  

#### **1. Setting Up the Environment**  

1. Create a virtual environment and activate it:  
   ```sh
   python -m venv caire
   source caire/bin/activate
   ```  
2. Install dependencies:  
   ```sh
   pip install -e .
   ```  

#### PyTorch Installation

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

For further details regarding the installation of PyTorch, refer to the [official PyTorch guide](https://pytorch.org/get-started/locally/).

**Note:**  
If you are using an **Ampere GPU**, ensure that your CUDA version is **11.7 or higher**, then install **FlashAttention** with the following command:  

```sh
pip install flash-attn --no-build-isolation
```

#### **2. Additional Setup Functionality**

- **Creates necessary directories**: Ensures the existence of required folders (data/, and src/outputs/).
- **Downloads dataset files (~31GB)**: Fetches preprocessed datasets and lookup files, storing them in data/.

#### Downloading Assets and Data

```sh
python setup.py download_assets
```

---

## **Usage**  

### **Running CAIRE**  

- CAIRE processes datasets stored as folders of images. 
- An example dataset with five images is provided in `src/examples`.  
- The dataset name and various configurations are specified in `src/config.py`, including the base path (`BP`) and important parameters for retrieval and model processing.  

### **Configuration Details (`config.py`)**  

- **Paths**  
  - `BP`, `DATA_PATH`, and `OUTPUT_PATH` define the locations for input data and outputs.  

- **Retrieval & Indexing**  
  - `INDEX_INFOS`, `FAISS_INDICES`, and `LEMMA_EMBEDS` are paths used for image and text retrieval metadata.  
  - `BABELNET_WIKI` stores Wikipedia sources of the BabelNet entities.  
  - `RETRIEVAL_BATCH_SIZE` controls the batch size for retrieval.  

- **Wikipedia Retrieval**  
  - `MAX_WIKI_DOCS` limits the number of Wikipedia documents retrieved per query image.  

- **Culture scoring**  
  - `PROMPT_TEMPLATE`
  - `TARGET_LIST` is the list of possible culture labels 

### **Running the Pipeline**  

This pipeline is split into two stages.
The first script processes images and retrieves relevant data, while the second computes cultural relevance scores.  

#### **Option 1: Running the Scripts Manually**  

```sh
python -m src.main_VEL      # Visual Entity Linking
python -m src.main_culture  # Cultural relevance scoring
```  

#### **Option 2: Running with a Shell Script**  

```sh
chmod +x src/run_pipeline.sh  
./src/run_pipeline.sh         
```  

Processed images and outputs will be saved in `src/outputs`.

---

### **Output Files**  

After running `src/main.py`, the following files will be created in `src/outputs/`:  

- **`{DATASET}_bids_match.pkl`** – Entity matching results  
- **`caire_{DATASET}_lemma_match.pkl`** – Lemma-based matching results  
- **`caire_{DATASET}_wiki.pkl`** – Wikipedia-based retrieval results  
- **`{DATASET}_image_embeddings.pkl`** – Image embeddings  
- **`1-5_{DATASET}_VLM_qwen.pkl`** – Final 1-5 scoring results  

### **Visualization**  

`eval/visualization.ipynb` shows the 1-5 scores and matched Wikipedia pages for the example images.

---
## Storage Requirements  

> [!IMPORTANT]
> Ensure you have sufficient disk space before proceeding:
- data/ requires **~31GB**