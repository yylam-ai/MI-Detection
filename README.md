# MI-Detection
This repository is modified based on [multiEchoAI](https://github.com/degerliaysen/MultiEchoAI) due to fixes on outdated libraries.

## Setup Guide

### 1. Initialize a Virtual Environment
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment
#### Windows:
```bash
venv\Scripts\activate
```

#### macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install Dependencies
Once the virtual environment is activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 4. Myocardial infarction detection by AI-models
The detection of myocardial infarction can be carried out for each AI-model with respect to the given echocardiography view as follows:
```
python train.py --view multi
python train.py --view 2CH
python train.py --view 4CH
```
To specify the GPUs in a server, the code also be executed as follows:
```
python train.py --gpu 0 --view multi
python train.py --gpu 1 --view 2CH
python train.py --gpu 2 --view 4CH
```

## Run in Google Colab
You can run this project in Google Colab by clicking this [link](https://colab.research.google.com/drive/1BTYrgHukwEkZd9Sp_czROb8YzuXFVk37?usp=sharing)

## Additional Notes
- Ensure you have Python installed before proceeding.
- Run `deactivate` to exit the virtual environment when done.

# Neural Architecture Search of Deep Priors (DP-NAS)
This work is based on [DP-NAS](https://github.com/ccc-frankfurt/DP-NAS), the implementation is  modified for MI-detection task.

### NAS for deep-prior
```
python dpnas/main.py --dataset MI -b 8 --patch-size 12 -pf 4 -lr 1e-4 --epochs 50
```

### NAS for CNN
```
python dpnas/main.py --dataset MI -b 8 --patch-size 12 -pf 4 -lr 1e-4 --epochs 50 --full-training true
```
### Training individual or a set of best architectures from the search
By default this will run the experiment, i.e. it will pick 6 low performing, 6 medium range and 6 top performing deep priors, resample the weights and train classifiers on top in order to gauge the variability when weights are re-sampled.
```
python dpnas/main.py -t 2 --dataset MI -b 8 --patch-size 12 -pf 4 -lr 1e-4 --epochs 50 --replay-buffer-csv-path <path>
```
After that, the best performing architecture may be selected manually and used to train the MI-dataset using K-fold cross-validation and evaluated accordingly.

