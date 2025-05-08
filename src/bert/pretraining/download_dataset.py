#!/usr/bin/env python3
import os
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import argparse
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = "datasets/pretraining"
OUTPUT_FILE = os.path.join(DATA_DIR, "data.txt")

def process_books_and_wikipedia(max_documents=2000):
    os.makedirs(DATA_DIR, exist_ok=True)

    half_max = max_documents // 2

    print("Loading BooksCorpus dataset...")
    try:
        books = load_dataset("bookcorpus", split="train", trust_remote_code=True)
    except Exception as e:
        print("Error loading 'bookcorpus':", e)
        print("Falling back to 'bookcorpusopen' dataset.")
        books = load_dataset("bookcorpusopen", split="train", trust_remote_code=True)
    
    print("Loading Wikipedia dataset...")
    wiki = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
    
    nltk.download("punkt")
    
    books_documents = []
    wiki_documents = []
    
    print("Processing BooksCorpus...")
    for sample in tqdm(books, desc="BooksCorpus processing"):
        text = sample.get("text", "")
        if not text:
            continue
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            continue
        doc = "\n".join(sentences)
        books_documents.append(doc)
        if len(books_documents) >= half_max:
            break

    print("Processing Wikipedia...")
    for sample in tqdm(wiki, desc="Wikipedia processing"):
        text = sample.get("text", "")
        if not text:
            continue
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            continue
        doc = "\n".join(sentences)
        wiki_documents.append(doc)
        if len(wiki_documents) >= half_max:
            break

    documents = books_documents + wiki_documents
    print(f"Total documents processed: {len(documents)}")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(documents))
    print(f"Processed dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process dataset for pretraining")
    parser.add_argument("--max_documents", type=int, default=80000,
                        help="Maximum number of documents to process (combined from both datasets)")
    args = parser.parse_args()
    
    process_books_and_wikipedia(max_documents=args.max_documents)