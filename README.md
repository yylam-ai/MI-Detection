# MI-Detection

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


