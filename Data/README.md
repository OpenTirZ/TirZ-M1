# Dataset

This directory contains the dataset used for training the **TirZ M1** language model.

## Data Source

The training text is collected from **Project Gutenberg**, a large collection of public-domain books and literary works.

Project Gutenberg: https://www.gutenberg.org/

## Data Collection

The dataset is downloaded from a hosted text file and processed using Python.

```python
import requests
from bs4 import BeautifulSoup

url = "https://raw.githubusercontent.com/spyguessgame-boop/own_dataset/refs/heads/main/data.txt"

response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")
text_data = soup.get_text()

print(text_data[:10000])
```

## Processing

The following preprocessing steps are applied:

1. Download the raw text data.
2. Extract plain text content.
3. Remove unnecessary HTML formatting (if present).
4. Prepare the text for tokenization and model training.

## Files

* `data.py` — Script used to download and preprocess the dataset.
* `README.md` — Documentation for the dataset.

## License

The original texts belong to their respective authors and are distributed through Project Gutenberg under Project Gutenberg's terms and conditions.

Users are responsible for ensuring compliance with the licenses and usage policies of the original data sources.

## Purpose

This dataset is used exclusively for training and experimentation of the TirZ M1 language model.
