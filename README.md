<div align="center">

# *CAIRE: Cultural Attribution of Images by Retrieval-Augmented Evaluation*

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
The setup script will:  
- Create necessary directories (`data/` and `checkpoints/`)  
- Download model checkpoints (~4GB)  
- Download dataset files (~31GB)  
- Set up a Conda environment (`caire`)  

Run:  
```sh
python setup.py
```  

### **Step 3: Activate the Conda Environment**  
```sh
conda activate caire
```  

## Storage Requirements  

Ensure you have enough disk space:  
- `checkpoints/` requires **~4GB**  
- `data/` requires **~31GB**  

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

## Project Structure  

```
📂 assets/            
📂 eval/                
  ├── 📂 outputs/             
  │   ├── 📄 1-5_examples.ipynb      # Jupyter Notebook for analysis  
📂 src/  
  ├── 📂 examples/                   # Sample images for testing  
  │   ├── 🖼️ eg1.jpg  
  │   ├── 🖼️ eg2.jpg  
  │   ├── 🖼️ eg3.jpg  
  │   ├── 🖼️ eg4.jpg  
  │   ├── 🖼️ eg5.jpg  
  ├── 📂 models/                     # Model-related scripts  
  │   ├── 📄 model_loader.py         # Model loading script  
  ├── 📂 scripts/                    # Core functionalities  
  │   ├── 📄 __init__.py  
  │   ├── 📄 culture_scores.py       # Cultural score calculations  
  │   ├── 📄 disambiguation.py       # Lemma matching logic  
  │   ├── 📄 fetch_wikipedia.py      # Wikipedia data retrieval  
  │   ├── 📄 retrieval.py            # Entity retrieval  
  │   ├── 📄 config.py               # Configuration settings  
  │   ├── 📄 main.py                 # Main script  
  │   ├── 📄 utils.py                # Utility functions  
📄 .gitignore                        # Git ignore file  
📄 README.md                         # Documentation  
📄 setup.py                          # Installation script  
```  
