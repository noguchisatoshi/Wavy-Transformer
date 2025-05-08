#!/usr/bin/env python3
import os
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import argparse
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = "./datasets/analyze_oversmoothing"
OUTPUT_FILE = os.path.join(DATA_DIR, "wikibio_data.txt")

def parse_max_documents(val):
    if isinstance(val, int):
        return val
    if val.lower() == "full":
        return None
    try:
        return int(val)
    except ValueError:
        raise argparse.ArgumentTypeError("max_documents must be an integer or 'full'.")

def process_wikibio(max_documents=2000):
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Loading michaelauli/wiki_bio dataset...")
    try:
        dataset = load_dataset("michaelauli/wiki_bio", split="train", trust_remote_code=True)
    except Exception as e:
        print("Error loading 'michaelauli/wiki_bio':", e)
        return

    nltk.download("punkt")
    documents = []
    
    print("Processing Wikibio dataset...")
    for sample in tqdm(dataset, desc="Wikibio processing"):
        text = sample.get("target_text", "")
        if not text:
            continue
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            continue
        doc = "\n".join(sentences)
        documents.append(doc)
        if max_documents is not None and len(documents) >= max_documents:
            break
    
    print(f"Total documents processed: {len(documents)}")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(documents))
    print(f"Processed dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process wiki_bio dataset for analyzing over smoothing")
    parser.add_argument("--max_documents", type=parse_max_documents, default="1000",
                        help="Maximum number of documents to process. Use 'full' to process all documents.")
    args = parser.parse_args()
    
    process_wikibio(max_documents=args.max_documents)