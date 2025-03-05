# CAIRE  

## Overview  

<p align="center">
  <img src="assets/fig.jpg" width="100%">
</p>  

CAIRE is a cultural-aware retrieval and evaluation framework designed for analyzing and processing multimodal data.  

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
  │   ├── 📄 1-5_examples.ipynb  # Jupyter Notebook for analysis  
📂 src/
  ├── 📂 examples/              # Sample images for testing  
  │   ├── 🖼️ eg1.jpg  
  │   ├── 🖼️ eg2.jpg  
  │   ├── 🖼️ eg3.jpg  
  │   ├── 🖼️ eg4.jpg  
  │   ├── 🖼️ eg5.jpg  
  ├── 📂 models/                # Model-related scripts  
  │   ├── 📄 model_loader.py     # Model loading script  
  ├── 📂 scripts/               # Core functionalities  
  │   ├── 📄 __init__.py  
  │   ├── 📄 culture_scores.py   # Cultural score calculations  
  │   ├── 📄 disambiguation.py   # Disambiguation logic  
  │   ├── 📄 fetch_wikipedia.py  # Wikipedia data retrieval  
  │   ├── 📄 retrieval.py        # Information retrieval  
  │   ├── 📄 config.py           # Configuration settings  
  │   ├── 📄 main.py             # Main script  
  │   ├── 📄 utils.py            # Utility functions  
📄 .gitignore                  # Git ignore file  
📄 pips.txt                     # List of dependencies  
📄 README.md                    # Documentation  
📄 setup.py                      # Installation script  
```  