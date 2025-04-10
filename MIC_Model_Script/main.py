import os
import re
import zipfile
import gdown
import pandas as pd
import nltk
import spacy
from datetime import datetime
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from word2number import w2n

# -----------------------------------
# Download pretrained model from Google Drive
# -----------------------------------

model_dir = "./model"
model_zip = "model.zip"

if not os.path.exists(model_dir):

    file_id = "110cqsqjlMuKvS5x50NTk4ZQsQGOf0FgJ" 
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print("📥 Downloading pretrained model...")
    gdown.download(url, model_zip, quiet=False)

    print("📦 Unzipping model...")
    with zipfile.ZipFile('model.zip', 'r') as zip_ref:
        zip_ref.extractall(model_dir)

    print("✅ Model ready.")

nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')

# -----------------------------------
# Text Cleaning Function
# -----------------------------------

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"--+|\.{2,}|\s?_\s?", " ", text)
    text = re.sub(r"\b(links|url)\b", "", text)

    stop_words = set(stopwords.words("english"))
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# -----------------------------------
# Preprocess Articles from Folder
# -----------------------------------

def process_articles(base_path, output_path):
    for year in range(2011, 2013):
        folder_path = os.path.join(base_path, str(year))
        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue

        year_folder = os.path.join(output_path, str(year))
        os.makedirs(year_folder, exist_ok=True)

        for filename in os.listdir(folder_path):

            if filename == ".ipynb_checkpoints": 
                continue

            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
                sections = re.split(r'-{5,}|_{5,}', content)
                filtered_sections = [s for s in sections if "Search Strategy" not in s]

                cleaned_articles = [clean_text(s) for s in filtered_sections if s.strip()]
                if cleaned_articles:
                    csv_filename = os.path.splitext(filename)[0] + ".csv"
                    output_file = os.path.join(year_folder, csv_filename)

                    df = pd.DataFrame({'year': [year] * len(cleaned_articles), 'text': cleaned_articles})
                    df.to_csv(output_file, index=False, encoding='utf-8')
                    print(f"✅ Saved: {output_file} ({len(df)} articles)")

# -----------------------------------
# Extractor Class
# -----------------------------------

class ConflictInfoExtractor:
    def __init__(self, model_path="./model", valid_states=None):
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.classifier = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )
        self.valid_states = set(state.lower() for state in valid_states) if valid_states else set()

    def clean_text(self, text):
        return str(text).replace("\n", " ").strip()

    def _format_date(self, date_str):
        for fmt in ["%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return date_str

    def _convert_to_number(self, word):
        try:
            return str(w2n.word_to_num(word))
        except:
            return word if word.isdigit() else ""

    def extract_info(self, text):
        text = self.clean_text(text)
        entities = self.classifier(text)

        result = {
            "date": "",
            "min_fatalities": "",
            "max_fatalities": "",
            "countries": []
        }
        date_parts = []

        for entity in entities:
            word = entity["word"].replace("#", "").strip(" ,.-").lower()
            if not word or word in [',', '-', '.', '']:
                continue

            label = entity["entity_group"]
            if label == "DATE":
                date_parts.append(word)
            elif label == "MIN_FAT":
                result["min_fatalities"] = self._convert_to_number(word)
            elif label == "MAX_FAT":
                result["max_fatalities"] = self._convert_to_number(word)
            elif label == "COUNTRY" and word in self.valid_states:
                result["countries"].append(word)

        full_date = " ".join(date_parts).strip()
        result["date"] = self._format_date(full_date)
        result["countries"] = sorted(set(result["countries"]))
        result["min_fatalities"] = result["min_fatalities"] or "0"
        result["max_fatalities"] = result["max_fatalities"] or "0"
        return result

    def process_csv(self, csv_path, text_column="text"):
        df = pd.read_csv(csv_path)
        df[text_column] = df[text_column].fillna("").astype(str)
        return pd.DataFrame([self.extract_info(text) for text in df[text_column]])

# -----------------------------------

def main():
    base_path = "./MIC_Dataset"          # Update this path with your dataset folder if needed
    output_path = "./processed_articles"  # Each article CSV will be stored here

    print("Preprocessing articles...")
    process_articles(base_path, output_path)

    print("Loading valid country names...")
    df_states = pd.read_csv("./Data/states2016.csv")
    valid_states = df_states["statenme"].dropna().str.lower().tolist()

    print("Running conflict info extractor...")
    extractor = ConflictInfoExtractor(model_path=model_dir, valid_states=valid_states)

    # Combine all yearly CSVs for processing
    combined_rows = []
    for year_folder in os.listdir(output_path):
        folder_path = os.path.join(output_path, year_folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                df = extractor.process_csv(file_path, text_column="text")
                df["source_file"] = file
                df["year"] = year_folder
                combined_rows.append(df)
            except Exception as e:
                print(f" Failed to process {file_path}: {e}")

    if combined_rows:
        final_df = pd.concat(combined_rows, ignore_index=True)
        final_df["countries"] = final_df["countries"].apply(lambda x: "; ".join(x) if isinstance(x, list) else x)
        final_df.to_csv("Extracted_info.csv", index=False)
        print("Output saved to Extracted_info.csv")

if __name__ == "__main__":
    main()