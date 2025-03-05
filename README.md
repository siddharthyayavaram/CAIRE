<div align="center">

# *CAIRE: Cultural Attribution of Images by Retrieval-Augmented Evaluation*

[The final directory structure after setup and running the pipeline](##final-Structure)

## Overview  

<p align="center">
  <img src="assets/fig.jpg" width="100%">
</p>  

</div>

## Abstract

As text-to-image models become increasingly prevalent, ensuring their equitable performance across diverse cultural contexts is critical. While prompt-based interventions have been explored to mitigate biases, they often introduce factual inaccuracies or offensive content. Despite widespread recognition of these challenges [¹](https://tinyurl.com/yc5jjk64), there is no reliable metric to evaluate **cultural relevance** in generated images.  
To address this gap, we introduce **CAIRE**, a novel framework that assesses cultural relevance by grounding images to entities and concepts in a knowledge base for a user-defined set of free-text labels. On a synthetically constructed dataset of rare and culturally significant items (*synthetic*), built using proprietary models, CAIRE surpasses all baselines by **28% F1 points**.  

Additionally, we evaluate text-to-image (T2I) models by:  
- Generating images for culturally universal concepts (*concept-generated*)  
- Retrieving real-world images of the same concepts (*concept-natural*)  

CAIRE achieves Pearson’s correlations of **0.56** and **0.66** with human ratings on these sets, based on a 5-point Likert scale of cultural relevance. This demonstrates strong alignment with human judgment across diverse image sources.  

## Installation & Setup  

Before running CAIRE, ensure you have Conda installed.  

### **Step 1: Clone the Repository**  
```sh
git clone https://github.com/siddharthyayavaram/CAIRE
cd CAIRE
```  

### **Step 2: Run the Setup Script**  
The setup script performs the following tasks:  
- **Creates necessary directories**: Ensures that the required folders (`data/`, `checkpoints/`, and `src/outputs/`) exist.  
- **Clones the `big_vision` repository**: Clones Google's [`big_vision`](https://github.com/google-research/big_vision) repository.  
- **Downloads model checkpoints (~4GB)**: Retrieves pre-trained model checkpoints from Google Cloud Storage and stores them in the `checkpoints/` directory.  
- **Downloads dataset files (~31GB)**: Fetches various preprocessed datasets and lookup files, storing them in `data/`.  
- **Sets up a Conda environment**: Creates a new conda environment `caire` from `environment.yaml`.

### **Next Steps**  
Once setup is complete, activate the Conda environment using:  
```bash
conda activate caire
```
Then proceed with running your models or experiments.

Run:  
```sh
python setup.py
```  

### **Step 3: Activate the Conda Environment**  
```sh
conda activate caire
```  

## Storage Requirements  

> [!IMPORTANT]
> Ensure you have sufficient disk space before proceeding:
- checkpoints/ requires **~4GB**
- data/ requires **~31GB**

## Usage  

Run CAIRE using:  
```sh
python -m src.main
```  
This processes the example images in `src/examples`.  
Results can be found in `eval/src/analysis.ipynb`.  

## Configuration  

Modify `config.py` to adjust runtime settings.  

## Notes  

- Ensure `gsutil` is installed and authenticated for required file access.  
- If setup fails, check your network connection and available storage before retrying.  

## Final Structure  

```
📂 CAIRE/                                              
│-- 📂 assets/                                
│-- 📂 eval/                                            # Evaluation-related files
│   ├── 📂 src/                                         # Evaluation scripts and notebooks
│   │   ├── 📄 analysis.ipynb                           # Jupyter Notebook for analysis
│   ├── 📂 outputs/                                     # Evaluation outputs
│   │   ├── 📄 1-5_examples_VLM_qwen.pkl                # Output for example images
│
│-- 📂 src/                                             # Source code directory
│   ├── 📂 examples/                                    # Sample images for testing (DATASET)
│   │   ├── 🖼️ eg1.jpg  
│   │   ├── 🖼️ eg2.jpg  
│   │   ├── 🖼️ eg3.jpg  
│   │   ├── 🖼️ eg4.jpg  
│   │   ├── 🖼️ eg5.jpg  
│   ├── 📂 models/                            
│   │   ├── 📄 model_loader.py                          # Model loading script  
│   ├── 📂 scripts/                                     # Core functionalities  
│   │   ├── 📄 culture_scores.py                        # Cultural score calculations  
│   │   ├── 📄 disambiguation.py                        # Lemma matching logic  
│   │   ├── 📄 fetch_wikipedia.py                       # Wikipedia data retrieval  
│   │   ├── 📄 retrieval.py                             # Entity retrieval
│   ├── 📂 outputs/                                     # Outputs from running the pipeline (main)
│   │   ├── 📄 {DATASET}_bids_match.pkl                 # Entity matching results
│   │   ├── 📄 caire_{DATASET}_lemma_match.pkl          # Lemma matching results
│   │   ├── 📄 caire_{DATASET}_wiki.pkl                 # Wikipedia-based retrieval results
│   │   │── 📄 {DATASET}_image_embeddings.pkl           # Image embeddings
│   │   │── 📄 1-5_{DATASET}_VLM_qwen.pkl               # Final 1-5 scores
│   ├── 📄 config.py                                    # Configuration settings  
│   ├── 📄 main.py                                      # Main script  
│   ├── 📄 utils.py                          
│
│-- 📂 checkpoints/                                     # Model checkpoints
│   ├── 📄 sentencepiece.model                
│   ├── 📄 webli_i18n_so400m_16_256_78061115.npz
│
│-- 📂 data/                                            # KB Metadata
│   ├── 📄 babelnet_source_dict.pkl                     # Wikipedia entities from BabelNet
│   ├── 📄 combined_lemma_embeds.pkl                    # Precomputed lemma embeddings
│   ├── 📂 faiss_index_merged/                          # FAISS index for retrieval
│   ├── 📄 index_infos_merged.json                      # Metadata corresponding to retrieval
│   ├── 📄 country_list.pkl                             # List of 10 populous, diverse countries
│
│-- 📂 big_vision/                                      # Cloned repository
│
│-- 📄 README.md                              
│-- 📄 setup.py                                         # Installation script  
│-- 📄 environment.yaml                                 # Conda environment file

```  
