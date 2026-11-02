"""
RAG Pipeline - Stage 1: Setup & Live Dataset Preview
----------------------------------------------------
Goal: 
  1. Set up our libraries, embedding model, and persistent ChromaDB storage.
  2. Connect to the live Hugging Face dataset stream and preview raw records
     in the terminal so we understand the exact structure of our source data.
"""

import json
import numpy as np
from sympy import true
import torch
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch.cpu import is_available

# =====================================================================
# STEP 1: Hardware & Model Setup 🧠
# =====================================================================
def initialize_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Detects the fastest chip on your computer and loads the embedding model.
    - 'cuda': NVIDIA GPU
    - 'mps':  Apple Silicon GPU (M1/M2/M3/M4/M5 Macs)
    - 'cpu':  Standard processor fallback
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else: 
        device = "cpu"
    
    print(f"[1/3] Loading embedding model '{model_name}' on device: [{device.upper()}]...")
    model = SentenceTransformer(model_name, device=device)
    print(f"    -> Model loaded successully!\n")
    return model

# =====================================================================
# STEP 2: Persistent Vector Database Setup 💾
# =====================================================================
def initialize_chroma_collection(
    persist_dir: str = "./chroma_financial_db",
    collection_name: str = "cfpb_disputes"
):
    """
    Creates (or connects to) a folder on your hard drive to store vectors permanently,
    and opens a collection (similar to a SQL table) configured for Cosine Similarity.
    """
    print(f"[2/3] Connecting to chromadb folder at '{persist_dir}'...")

    # PersistentClient saves all data to disk so you don't lose vectors when the script ends
    client = chromadb.PersistentClient(path=persist_dir)

    # get_or_create_collection opens the table if it exists, or creates it if it's new.
    # metadata={"hnsw:space": "cosine"} tells ChromaDB to use Cosine Similarity for search.
    collection = client.get_or_create_collection(
        name = collection_name,
        metadata = {"hnsw:space":"cosine"}
    )
    print(f"  -> Collection '{collection_name}' ready! (current records saved: {collection.count()}) \n")
    return collection

# =====================================================================
# STEP 3: Developer Data Preview Function 🔍
# =====================================================================

def preview_hf_dataset(
    dataset_name: str = "claritystorm/cfpb-consumer-complaints",
    num_samples: int = 2
):
    """
    Streams rows from a live Hugging Face dataset and prints the column schema
    plus 'num_samples' non-empty records so we can inspect the raw data during development.
    """
    print(f"[3/3] Connecting to live Hugging Face dataset: '{dataset_name}'...")

    # streaming=True fetches records one-by-one over HTTP without downloading the whole dataset
    dataset_stream = load_dataset(dataset_name, split="train", streaming="True")
    print("         -> Connection established! Inspecting raw data structure.. \n")
    print("=" * 75)
    print("Raw hugging face dataset preview")
    print("=" * 75)

    samples_shown = 0
    for row_index, row in enumerate(dataset_stream):
        # On the very first row, print all available column names (dictionary keys)
        if row_index == 0:
            print(f"Available Columns ({len(row.keys())} total):")
            print(f"  {list(row.keys())} \n")

        # Grab the customer's written complaint text (some rows in CFPB data are blank/None)
        narrative = row.get("consumer_complaint_narrative") or row.get("consumer_narrative")

        # Skip rows where the customer didn't write a narrative story
        if not narrative or len(str(narrative).strip()) < 40:
            continue
            
        samples_shown += 1
        print(f"--- Raw Sample Record #{samples_shown} (Stream Row Index: {row_index}) ---")
        # Pretty-print the dictionary fields so we can see what metadata is available
        for key, value in row.items():
            # Truncate very long text fields just for a clean console preview
            display_val = str(value)[:120] + "..." if value and len(str(value)) > 120 else value
            print(f"  .{key:<30}: {display_val}")
        print()

        # Stop streaming once we've previewed enough valid samples
        if samples_shown >= num_samples:
            break
    print("=" * 75 + "\n")

# =====================================================================
# STEP 4: Sliding-Window Text Chunker ✂️
# =====================================================================

def chunk_text(text: str, chunk_size: int=500, overlap: int = 75) -> list:
    """
    Splits a long string into smaller overlapping pieces.
    - chunk_size (500): Maximum characters per slice (~80-100 words).
    - overlap (75): Characters shared between back-to-back slices so sentences aren't cut in half.
    """
    # Clean out extra whitespace/newlines first
    clean_text = " ".join(str(text).split())

    # If the complaint is already shorter than 500 characters, return it as a single chunk
    if not clean_text or len(clean_text) <= chunk_size:
        return [clean_text] if clean_text else []

    chunks = []
    start = 0

    # Slide a 500-character window across the text until we reach the end
    while start < len(clean_text):
        end = start + chunk_size
        chunk_slice = clean_text[start:end].strip()
        chunks.append(chunk_slice)
    
    # If our window reached or passed the end of the text, we are done
        if end >= len(clean_text):
            break

    # Move the starting point forward, but step back by 'overlap' 
    start += (chunk_size - overlap)

    return chunks 

# =====================================================================
# STEP 5: Developer Preview — Chunking & Metadata Enrichment 🔍
# =====================================================================

def preview_chunking_pipeline(
            dataset_name: str = "claritystorm/cfpb-consumer-complaints",
            chunk_size: int = 500,
            overlap: int = 75
        ):

            """
            Finds 1 long complaint (>600 chars) from Hugging Face, slices it into overlapping chunks,
            attaches metadata, and prints the exact 'Before vs. After' transformation in the console.
            """
            print("[4/4] Testing Chunking and Enrichment on a live multi-chunk complaint...")
            dataset_stream = load_dataset(dataset_name, split="train", streaming="true")

            for row in dataset_stream:
                narrative = row.get("consumer complaint narrative") or row.get("consumer_narrative")

                # Specifically grab a complaint longer than 600 chars so we can see 2+ chunks and their overlap
                if not narrative or len(str(narrative).strip()) <= 600:
                    continue

                # 1. Extract structured fields (providing safe fallbacks if a field is missing)
                complaint_id = str(row.get("complaint_id") or "Unknown_id")
                company = str(row.get("company") or "Unknown_company")
                product = str(row.get("product") or "financial Product")
                issue = str(row.get("issue") or "General Dispute")
                status = str(row.get("company_response_to_consumer") or row.get("comapany_response") or "pending")
                state = str(row.get("state") or "N/A")

                # 2. Slice the raw narrative into overlapping chunks
                raw_chunks = chunk_text(narrative, chunk_size=chunk_size, overlap=overlap)

                print("\n" + "=" * 80)
                print("stage 2 preview before and after chunking & enrichment")
                print("=" * 80)
                print(f"Original Complaint ID: {complaint_id}")
                print(f"Original Char length: {len(narrative)}characters")
                print(f"Total Chunks Created: {len(raw_chunks)} chunks (chunk_size={chunk_size}, overlap={overlap}) \n")

                # 3. Inspect each chunk and show how we package it for ChromaDB
                for chunk_idx, chunk_slice in enumerate(raw_chunks):
                    # Unique ID for every single chunk (Complaint ID + Chunk Number)
                    record_id = f"cfpb_{complaint_id}_c{chunk_idx}"

                    # Stamp the context badge onto the front of the text chunk
                    enriched_text = (
                            f"[Dispute ID: {complaint_id} | Company: {company} |"
                            f"Product: {product} | Issue: {issue} | Status: {status}]"
                            f"Narrative: {chunk_slice}"
                    )

                    # Separate dictionary of tags that ChromaDB can filter on (like SQL WHERE clauses)
                    metadata = {
                        "complaint_id": complaint_id,
                        "company": company,
                        "product": product,
                        "issue": issue,
                        "status": status,
                        "state": state,
                        "chunk_idx": chunk_idx
                    }

                    print(f" --- Chunk #{chunk_idx} (Record ID: {record_id}) ---")
                    print(f" .Metadata Dict: {metadata}")
                    print(f" .Raw Slice: \"{chunk_slice}\"")
                    print(f" .Enriched Text: \"{enriched_text[:160]}...\"")

                # Highlight the exact 75-character overlap between Chunk 0 and Chunk 1
                if len(raw_chunks) >= 2:
                    print(" Overlap Verification")
                    print(f" End of chunk 0 (last 75 characters) : \"...{raw_chunks[0][-overlap:]}\"")
                    print(f" Start of Chunk #1 (first 75 characters) : \"...{raw_chunks[1][:overlap]}...\"")

                print("=" * 80 + "\n")
                break # Stop after inspecting 1 full example

# =====================================================================
# MAIN EXECUTION (Testing Stage 1) 🚀
# =====================================================================
if __name__ == "__main__":
    # 1. Load the embedding model
    encoder = initialize_embedding_model()

    # 2. Connect to our local ChromaDB storage
    chroma_collection = initialize_chroma_collection()

    # 3. Preview 2 raw records from Hugging Face to see what we are working with
    #preview_hf_dataset(num_samples=2)

    # Stage 2: Test Chunking & Metadata Enrichment
    preview_chunking_pipeline(chunk_size= 500, overlap= 75)
