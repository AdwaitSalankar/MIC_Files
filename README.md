# Coding Actions and Fatalities in Militarized Interstate Confrontations (MICs)

This project extracts structured information from news articles related to **(MIC)** — including **event date**, **countries involved**, and **fatality counts** — using a fine-tuned BERT model.

The `MIC_Model_Script` directory contains the `main.py` file, which serves as the main script.

To test the model on different data, simply replace the `MIC_Dataset` folder with your own data organized in the same year-based folder structure.

Alternatively, you can use this `Colab notebook` for a no-setup run: [Colab Notebook](https://colab.research.google.com/drive/1GOfUFwdlnhcXQ0e1cVGIi_i-kpCBP78-?usp=sharing)

---

## Folder Structure

    ipynb files/                   # Code files used in creating and training models
    ├── Cleaning_&_Preprocessing.ipynb    
    └── Training_DistilBERT.ipynb

    MIC_Model_Script/
    ├── Country_Names/
    |   └── states2016.csv          # Country/state reference file with abbreviations and name
    ├── MIC_Dataset/                # Folder containing year-wise news articles
    │   ├── 2011/
    │   ├── 2012/
    ├── main.py                     # Main script to run the extraction pipeline  
    └── requirements.txt            # List of Python dependencies  

---

## Fine-tuned Model
The fine-tuned model used for extracting conflict information is on Google Drive. It will be automatically downloaded and extracted during the first run of the script.

Link:
[Google Drive](https://drive.google.com/drive/folders/17zm1pC4VyzDt-sKau_O4rJio6q1ENtlh?usp=sharing)

---

### How to Run

```bash
cd MIC_Model_Script
```

Install dependencies with:
```bash
pip install -r requirements.txt
```
After installation:
```bash
python -m spacy download en_core_web_sm
```

```bash
python main.py
```
The script will:
1. Download and unzip the model
2. Clean and preprocess articles
3. Extract relevant MIC information
4. Save results to csv
5. No user input required!
