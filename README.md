# CAIRE

## Overview

<p align="center">
  <img src="assets/fig.jpg" width="100%">
</p>

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
- Download model checkpoints (~4GB)
- Download dataset files (~31GB)
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
- `checkpoints/` requires **~4GB**
- `data/` requires **~31GB**

## Usage
After setup, run the project using:
```sh
python -m src.main
```

This will run the code for the 5 example images in `src/examples`

## Configuration
Modify `config.py` to adjust settings.

## Notes
- Ensure that `gsutil` is installed and authenticated to access the required files.
- If the setup script fails at any step, verify your network connection and storage availability before re-running it.

## Project Structure

