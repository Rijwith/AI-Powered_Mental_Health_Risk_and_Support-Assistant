import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)  # ✅ Add labels
        return item

    def __len__(self):
        return len(self.labels)


class DistressDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


def prepare_datasets(df, tokenizer, max_len=128, test_size=0.2):
    """
    df: pandas DataFrame with columns:
        ['statement','status','cleaned_text','Normal','Depression','Suicidal','Anxiety','Bipolar','labels']
    tokenizer: HuggingFace tokenizer (e.g., RobertaTokenizer)
    """

    # Features (use cleaned_text)
    texts = df["cleaned_text"].astype(str).tolist()

    # Labels (use only the 5 binary columns)
    labels = df[["Normal", "Depression", "Suicidal", "Anxiety", "Bipolar"]].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Tokenize
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=max_len)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=max_len)

    # Build Dataset objects
    train_dataset = DistressDataset(train_encodings, y_train)
    test_dataset = DistressDataset(test_encodings, y_test)

    return train_dataset, test_dataset
