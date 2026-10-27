"""
Minimalist Retrieval-Augmented Generation (RAG) Pipeline
--------------------------------------------------------
This script demonstrates the core mechanics of a RAG system from scratch without 
relying on external vector database abstractions (like Pinecone or Chroma). 
It covers:
  1. Model Initialization
  2. Raw Data Preparation
  3. Ingestion & Vector Indexing (with developer inspection logs)
  4. Cosine Similarity Search & Retrieval (with score logging)
  5. Context-Augmented Prompt Construction
"""
import numpy as np
from sentence_transformers import SentenceTransformer

# =====================================================================
# 1. Initialize our Embedding Model 🧠
# =====================================================================
# 'all-MiniLM-L6-v2' is a lightweight, fast transformer model from Hugging Face.
# It maps sentences/paragraphs into a 384-dimensional dense vector space where 
# semantically similar texts are positioned close to one another.

print("[INIT] Loading embedding model 'all-MiniLM-L6-v2'...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
print("[INIT] Model loaded successully.\n")

# =====================================================================
# 2. Sample Financial Data (e.g., Stripe Dispute Logs) 📄
# =====================================================================
# In a production application, this data would be loaded from PDFs, SQL 
# databases, or APIs, and split into smaller "chunks" so each piece focuses 
# on a single coherent record or concept.

financial_logs = [
    "Dispute ID: dp_01. Status: Won. Amount: $150. Reason: Fraudlent_charge. Customer provided evidence.",
    "Dispute ID: dp_105. Status: Won. Amount: $160. Reason: Fraudulent charge. Customer provided evidence of delivery.",
    "Dispute ID: dp_102. Status: Lost. Amount: $45. Reason: Product not received. Merchant failed to provide tracking.",
    "Dispute ID: dp_103. Status: Pending. Amount: $900. Reason: Unrecognized. Awaiting bank review.",
]

# =====================================================================
# 3. Ingestion & Indexing Pipeline 📥
# =====================================================================
# We use a standard Python list of dictionaries as an in-memory vector database.
# Each entry stores:
#   - "id": A unique identifier for tracking the record
#   - "text": The original human-readable string (needed later for the LLM prompt)
#   - "vector": The 384-dimensional NumPy array (used for mathematical search)

vector_database = []
print("--- [STEP 3: INGESTION & INDEXING] ---")
for idx, chunk in enumerate(financial_logs):
  # encoder.encode() tokenizes the text, passes it through the transformer
  # layers, and pools the token outputs into a single 1D NumPy array of shape (384,).
  embedding = encoder.encode(chunk)

  # store the raw text alongside its nuclear vector representation
  record = {
        "id": f"chunk_{idx}",
        "text": chunk,
        "vector": embedding
  } 
  vector_database.append(record)

# --- DEVELOPER INSPECTION: View the Vector Database after Ingestion ---
# Printing an entire 384-float array per document clutters the console, 
# so we inspect the metadata, vector shape, data type, and the first 5 dimensions.

print(f"Total records ingested:{len(vector_database)}")
print(f"Vector shape per record: {vector_database[0]['vector'].shape} (384 dimensions)")
print(f"Vector data type:  {vector_database[0]['vector'].dtype}\n")

# this command below gives all the 384 values for embedding which is not presentable --
# print(f"values inside vector database{vector_database}")

print("Preview of 'vector database' contents:")
for record in vector_database:
  # slicing the first five numbers and then roundind them 
  vector_preview = np.round(record["vector"][:5], 4).tolist() 
  print(f" Record [{record['id']}]:")
  print(f" |---- Text:   \"{record['text']}\"")
  print(f" |---- Vector: {vector_preview} .... (379 more values)\n")

# =====================================================================
# 4. Query & Retrieval Pipeline ⚡
# =====================================================================

def retrieve_context(query: str, db: list, top_k: int = 1) -> list:
  """
    Finds the top_k most semantically relevant text chunks for a user query.

    Args:
        query (str): The natural language question from the user.
        db (list): The in-memory vector database containing 'text' and 'vector' keys.
        top_k (int): The number of highest-scoring text chunks to return.

    Returns:
        list: A list of the top_k matching text strings, ordered from most to least relevant.
    """
    # Convert the user's question into the exact same 384-dimensional vector space
    # so we can directly compare it against our stored document vectors.

  query_vector = encoder.encode(query)
  print(f"Query vector preview (first 5 dims) : {np.round(query_vector[:5], 4).tolist()}\n")

  scores = []
  print("calculating cosine similarity against each database record:")
  for item in db:
    # Cosine Similarity measures the cosine of the angle between two vectors.
    # Formula: (A · B) / (||A|| * ||B||)
    # Score ranges from -1.0 (opposite) to 1.0 (identical meaning).

    # 1. Dot product (A · B): Sums the element-wise multiplication of both vectors
    dot_product = np.dot(query_vector, item['vector'])

    # 2. Euclidean norms (||A|| and ||B||): Calculate the magnitude (length) of each vector
    norm_a = np.linalg.norm(query_vector)
    norm_b = np.linalg.norm(item['vector'])

    # 3. Divide dot product by the product of magnitudes to normalize for length
    similarity = dot_product / (norm_a * norm_b)
    scores.append(similarity)

    # Log the similarity score for each chunk so we can verify retrieval accuracy
    print(f"  -> [{item['id']}] Score: {similarity:.4f} | Text: \"{item['text'][:55]}...\"")

  # np.argsort(scores) returns indices that would sort the array in ASCENDING order (lowest to highest).
  # [::-1] reverses the array to DESCENDING order (highest similarity first).
  # [:top_k] slices only the top N indices we want.
  top_indices = np.argsort(scores)[::-1][:top_k]

  print(f"\n Winning record index: {top_indices.tolist()} (Top-{top_k})\n")

  # Extract and return the raw text strings corresponding to the winning indices
  return [db[i]["text"] for i in top_indices]

# =====================================================================
# 5. Execution 🚀
# =====================================================================

user_query = "What happened to the $45 charge"

# Retrieve the most relevant context chunk(s) from our vector database
retrieved_chunk = retrieve_context(user_query, vector_database, top_k=1)

# Join multiple chunks with a newline if top_k is greater than 1
context_string = "\n".join(retrieved_chunk)

# Construct the final grounded prompt for the LLM.
# Instructing the LLM to use ONLY the provided context prevents hallucinations.

system_prompt = f"""
You are a helpful financial assistant. Answer the user's question using ONLY the provided context.

Context:
{context_string}

User Question: {user_query}
"""

print("--- [STEP 5: FINAL LLM PROMPT] ---")
print(system_prompt)

