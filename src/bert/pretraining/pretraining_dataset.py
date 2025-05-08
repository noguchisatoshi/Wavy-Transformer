#!/usr/bin/env python3
# pretraining_dataset.py

import os
import random
import pickle
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast  # 高速トークナイザー
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import sys

logger = logging.getLogger("PretrainingDataset")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


class BertPretrainingDataset(Dataset):
    def __init__(self,
                 file_path,
                 tokenizer: BertTokenizerFast,
                 max_seq_length=512,
                 short_seq_prob=0.1,
                 masked_lm_prob=0.15,
                 max_predictions_per_seq=20,
                 dupe_factor=3,
                 nsp_neg_prob=0.5,
                 max_workers=64,
                 persistent_cache_path=None,
                 max_documents=None):

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.short_seq_prob = short_seq_prob
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.dupe_factor = dupe_factor
        self.nsp_neg_prob = nsp_neg_prob
        self.max_workers = max_workers

        if persistent_cache_path is not None and os.path.exists(persistent_cache_path):
            with open(persistent_cache_path, "rb") as f:
                loaded = pickle.load(f)
            cached_max_docs = loaded.get("max_documents", None)
            if max_documents != cached_max_docs:
                logger.warning(f"Cached max_documents ({cached_max_docs}) does not match specified max_documents ({max_documents}). Recomputing preprocessing.")
                self._process_and_cache(file_path, persistent_cache_path, max_documents)
            else:
                logger.info(f"Loading preprocessed dataset from {persistent_cache_path}")
                self.instances = loaded["instances"]
                logger.info(f"Loaded {len(self.instances)} instances from cache.")
        else:
            self._process_and_cache(file_path, persistent_cache_path, max_documents)

        self.total_instances = len(self.instances)

    def _process_and_cache(self, file_path, persistent_cache_path, max_documents):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        documents = [doc.splitlines() for doc in raw_text.split("\n\n") if doc.strip()]
        self.documents = documents
        self.all_sentences = [sent for doc in self.documents for sent in doc]

        self.instances = []
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        for dup in range(self.dupe_factor):
            if local_rank == 0:
                logger.info(f"Processing dupe_factor iteration {dup+1}/{self.dupe_factor}")
                sys.stdout.flush()
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        BertPretrainingDataset._process_doc,
                        doc,
                        self.tokenizer,
                        self.all_sentences,
                        self.max_seq_length,
                        self.short_seq_prob,
                        self.masked_lm_prob,
                        self.max_predictions_per_seq,
                        self.nsp_neg_prob,
                        dup
                    )
                    for doc in self.documents
                ]
                doc_count = 0
                total_docs = len(self.documents)
                next_log_percent = 1
                for future in as_completed(futures):
                    self.instances.extend(future.result())
                    doc_count += 1
                    progress_percent = (doc_count / total_docs) * 100
                    if local_rank == 0 and progress_percent >= next_log_percent:
                        logger.info(
                            f"Dupe iteration {dup+1}: {progress_percent:.1f}% complete "
                            f"({doc_count}/{total_docs} documents processed)"
                        )
                        sys.stdout.flush()
                        next_log_percent += 1

        random.shuffle(self.instances)
        if persistent_cache_path is not None:
            logger.info(f"Saving preprocessed dataset to {persistent_cache_path}")
            with open(persistent_cache_path, "wb") as f:
                pickle.dump({"instances": self.instances, "max_documents": max_documents}, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saving completed.")

    def __len__(self):
        return self.total_instances

    def __getitem__(self, idx):
        return self.instances[idx]

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
        while len(tokens_a) + len(tokens_b) > max_num_tokens:
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        return tokens_a, tokens_b

    @staticmethod
    def create_masked_lm_predictions(tokens, tokenizer, masked_lm_prob, max_predictions_per_seq):
        cand_indices = [i for i, token in enumerate(tokens) if token not in ("[CLS]", "[SEP]")]
        num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
        random.shuffle(cand_indices)
        masked_lm_labels = [-100] * len(tokens)
        masked_indices = sorted(cand_indices[:num_to_mask])
        for idx in masked_indices:
            original_token = tokens[idx]
            rand_val = random.random()
            if rand_val < 0.8:
                tokens[idx] = "[MASK]"
            elif rand_val < 0.9:
                tokens[idx] = original_token
            else:
                tokens[idx] = random.choice(list(tokenizer.vocab.keys()))
            masked_lm_labels[idx] = tokenizer.convert_tokens_to_ids(original_token)
        return tokens, masked_lm_labels

    @staticmethod
    def pad_sequence(input_ids, segment_ids, masked_lm_labels, max_seq_length, tokenizer):
        attention_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        segment_ids = segment_ids + [0] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        masked_lm_labels = masked_lm_labels + [-100] * padding_length
        return input_ids, segment_ids, attention_mask, masked_lm_labels

    @staticmethod
    def _process_doc(doc, tokenizer, all_sentences, max_seq_length,
                     short_seq_prob, masked_lm_prob, max_predictions_per_seq, nsp_neg_prob, dup):
        doc_seed = hash("".join(doc) + str(dup)) % (2**32)
        random.seed(doc_seed)

        instances = []
        if len(doc) < 2:
            return instances

        encoded = tokenizer.batch_encode_plus(
            doc,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_tensors=None
        )
        tokenized_sentences = [tokenizer.convert_ids_to_tokens(ids) for ids in encoded['input_ids']]

        target_seq_length = max_seq_length - 3
        if random.random() < short_seq_prob:
            target_seq_length = random.randint(2, target_seq_length)

        current_chunk = []
        current_length = 0
        for i, tokens in enumerate(tokenized_sentences):
            current_chunk.append(tokens)
            current_length += len(tokens)
            if i == len(tokenized_sentences) - 1 or current_length >= target_seq_length:
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)
                    tokens_a = []
                    for j in range(a_end):
                        tokens_a += current_chunk[j]
                    tokens_b = []
                    is_random_next = False
                    if len(current_chunk) == 1 or random.random() < nsp_neg_prob:
                        is_random_next = True
                        tokens_b = tokenizer.tokenize(random.choice(all_sentences))
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b += current_chunk[j]
                    tokens_a, tokens_b = BertPretrainingDataset._truncate_seq_pair(tokens_a, tokens_b, target_seq_length)
                    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
                    tokens, masked_lm_labels = BertPretrainingDataset.create_masked_lm_predictions(
                        tokens, tokenizer, masked_lm_prob, max_predictions_per_seq)
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    input_ids, segment_ids, attention_mask, masked_lm_labels = BertPretrainingDataset.pad_sequence(
                        input_ids, segment_ids, masked_lm_labels, max_seq_length, tokenizer)
                    nsp_label = 1 if is_random_next else 0
                    instance = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                        "token_type_ids": torch.tensor(segment_ids, dtype=torch.long),
                        "masked_lm_labels": torch.tensor(masked_lm_labels, dtype=torch.long),
                        "next_sentence_label": torch.tensor(nsp_label, dtype=torch.long)
                    }
                    instances.append(instance)
                current_chunk = []
                current_length = 0
        return instances