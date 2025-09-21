import os
import pandas as pd

# ----------------------------
# Dataset 1: Mental Health Conditions (multi-label)
# ----------------------------
def preprocess_dataset1(raw_path, processed_path):
    """
    Preprocess Dataset 1 (multi-label mental health conditions)
    and save a cleaned CSV.
    """
    df = pd.read_csv(raw_path)
    
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Ensure 'cleaned_text' exists
    if 'cleaned_text' not in df.columns:
        raise ValueError("Column 'cleaned_text' not found in Dataset 1")
    
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Dataset 1 processed CSV saved to: {processed_path}")


# ----------------------------
# Dataset 2: Emotion Dataset (single-label, train/test/val)
# ----------------------------
def preprocess_dataset2(txt_folder, processed_folder):
    """
    Preprocess Dataset 2 (emotion dataset with train/test/val txt files)
    and save cleaned CSVs with integer labels.
    """
    os.makedirs(processed_folder, exist_ok=True)
    splits = ['train', 'test', 'val']
    label_set = set()

    # First pass: create cleaned CSVs and collect labels
    for split in splits:
        txt_path = os.path.join(txt_folder, f"{split}.txt")
        df = pd.read_csv(txt_path, sep=';', header=None, names=['text', 'emotion'])
        
        df['text'] = df['text'].str.strip().str.lower()
        label_set.update(df['emotion'].unique())
        
        df.to_csv(os.path.join(processed_folder, f"{split}_clean.csv"), index=False)
        print(f"Dataset 2 {split} cleaned CSV saved.")

    # Create label mapping
    label2id = {label: i for i, label in enumerate(sorted(label_set))}
    id2label = {i: label for label, i in label2id.items()}

    # Second pass: add integer labels
    for split in splits:
        csv_path = os.path.join(processed_folder, f"{split}_clean.csv")
        df = pd.read_csv(csv_path)
        df['label'] = df['emotion'].map(label2id)
        df.to_csv(csv_path, index=False)

    print("Dataset 2 all splits processed with integer labels.")
    return label2id, id2label


# ----------------------------
# Dataset 3: Survey / Statistical Data
# ----------------------------
def preprocess_dataset3(raw_path, processed_path):
    """
    Preprocess Dataset 3 (survey / statistical data)
    for Phase 3 dashboards.
    """
    df = pd.read_csv(raw_path)
    
    # Keep only useful columns
    keep_cols = ['Indicator', 'Group', 'State', 'Subgroup', 'Value']
    df = df[keep_cols]
    
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Dataset 3 processed CSV saved to: {processed_path}")


# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    # Dataset 1
    preprocess_dataset1(
        raw_path="data/dataset1_conditions/raw/multilabled_preprocessed.csv",
        processed_path="data/dataset1_conditions/processed/multilabled_clean.csv"
    )

    # Dataset 2
    preprocess_dataset2(
        txt_folder="data/dataset2_emotions/raw",
        processed_folder="data/dataset2_emotions/processed"
    )

    # Dataset 3
    preprocess_dataset3(
        raw_path="data/dataset3_survey/raw/survey.csv",
        processed_path="data/dataset3_survey/processed/survey_clean.csv"
    )
