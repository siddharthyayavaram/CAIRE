# CAIRE

## Overview
CAIRE

<p align="center">
  <img src="assets/fig2.pdf" width="70%">
</p>

## Project Structure
```
/src
│── models/
│   │── model_loader.py   
│── scripts/
│   │── __init__.py       
│   │── culture_scores.py
│   │── disambiguation.py
│   │── fetch_wikipedia.py
│   │── retrieval.py      
│── __init__.py           
│── config.py             
│── main.py               
│── utils.py              
```

## Installation & Setup
Before running CAIRE, ensure that you have Conda installed.

### **Step 1: Clone the Repository**
```sh
git clone https://github.com/siddharthyayavaram/CAIRE
cd CAIRE
```

### **Step 2: Run the Setup Script**
Execute the following command to:
- Create necessary directories (`data/` and `checkpoints/`)
- Download model checkpoints (~5GB)
- Download dataset files (~30GB)
- Create a Conda environment (`caire`)

```sh
python setup.py
```

### **Step 3: Activate the Conda Environment**
Once setup is complete, activate the environment:
```sh
conda activate caire
```

## Storage Requirements

> [!IMPORTANT]
> Ensure you have sufficient disk space before proceeding:
- `checkpoints/` requires **~5GB**
- `data/` requires **~30GB**

## Usage
After setup, run the project using:
```sh
python -m src.main
```

## Configuration
Modify `config.py` to adjust settings for model behavior, data paths, or other configurations as needed.

## Notes
- Ensure that `gsutil` is installed and authenticated to access the required files.
- If the setup script fails at any step, verify your network connection and storage availability before re-running it.

