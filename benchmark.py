#!/usr/bin/env python3
"""
ChromaDB vs PGVector Benchmark using LangChain
This script benchmarks ChromaDB against PGVector across various scenarios:
1. Insertion performance
2. Query performance (similarity search)
3. Memory usage
4. CPU usage

Results are saved as CSV files for each benchmark scenario.
"""

import os
import time
import psutil
import numpy as np
import pandas as pd
import subprocess  # Add this import
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# LangChain vector stores
from langchain_chroma import Chroma
from langchain_postgres import PGVector

# Create output directory for results
os.makedirs("benchmark_results", exist_ok=True)

# Custom embedding class to use our pre-computed embeddings
class PrecomputedEmbeddings(Embeddings):
    def __init__(self, embeddings_dict):
        self.embeddings_dict = embeddings_dict

    def embed_documents(self, texts):
        return [self.embeddings_dict.get(text, [0] * 384) for text in texts]

    def embed_query(self, text):
        return self.embeddings_dict.get(text, [0] * 384)

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
VECTOR_DIM = 384  # Dimension of embeddings from the model

# Connection parameters
CHROMA_HOST = 'localhost'
CHROMA_PORT = 8009
PG_CONNECTION = "postgresql+psycopg://admin:admin123@localhost:5433/mydatabase"

# Test parameters
DATASET_SIZES = [100, 1000, 10000]  # Number of documents to test with
NUM_QUERIES = 50  # Number of queries to run for each test
BATCH_SIZES = [1, 10, 100, 1000]  # Batch sizes for batch operation tests

def get_test_data(size):
    """Generate test data of specified size"""
    newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    texts = newsgroups.data[:size]
    # Ensure we have enough data
    while len(texts) < size:
        texts.extend(texts[:size-len(texts)])
    texts = texts[:size]  # Trim to exact size

    # Generate embeddings
    embeddings = []
    for i in tqdm(range(0, len(texts), 100), desc="Generating embeddings"):
        batch = texts[i:i+100]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)

    # Create documents
    documents = []
    embeddings_dict = {}
    for i in range(size):
        doc = Document(
            page_content=texts[i],
            metadata={"id": i}
        )
        documents.append(doc)
        embeddings_dict[texts[i]] = embeddings[i]

    # Create embeddings class
    embedding_func = PrecomputedEmbeddings(embeddings_dict)

    return documents, embedding_func, embeddings_dict

def setup_chroma(embedding_func):
    """Set up ChromaDB client and collection"""
    try:
        # Reset collection if it exists
        import chromadb
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        try:
            client.delete_collection("benchmark_collection")
        except:
            pass

        # Create LangChain Chroma instance
        chroma_db = Chroma(
            collection_name="benchmark_collection",
            embedding_function=embedding_func,
            client=client
        )

        return chroma_db
    except Exception as e:
        print(f"Error setting up ChromaDB: {e}")
        return None

def setup_pgvector(embedding_func):
    """Set up PGVector connection and table"""
    try:
        # Create LangChain PGVector instance
        collection_name = "benchmark_collection"

        # Drop existing collection if it exists
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5433,
            database="mydatabase",
            user="admin",
            password="admin123"
        )
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS langchain_{collection_name}")
        cursor.execute(f"DROP TABLE IF EXISTS langchain_{collection_name}_embedding_metadata")
        conn.close()

        # Create new PGVector instance
        pgvector_db = PGVector(
            collection_name=collection_name,
            connection=PG_CONNECTION,
            embeddings=embedding_func,
            use_jsonb=True
        )

        return pgvector_db
    except Exception as e:
        print(f"Error setting up PGVector: {e}")
        return None

def benchmark_insertion(dataset_size):
    """Benchmark insertion performance"""
    documents, embedding_func, _ = get_test_data(dataset_size)
    results = []

    # ChromaDB insertion - handle batch size limit
    chroma_db = setup_chroma(embedding_func)
    if chroma_db:
        start_time = time.time()
        # Process in batches of 1000 to avoid exceeding ChromaDB's limit
        for i in range(0, len(documents), 1000):
            end_idx = min(i + 1000, len(documents))
            chroma_db.add_documents(documents[i:end_idx])
        chroma_time = time.time() - start_time
        results.append({"database": "ChromaDB", "operation": "insertion",
                       "dataset_size": dataset_size, "time_seconds": chroma_time})

    # PGVector insertion - also process in batches for consistency
    pgvector_db = setup_pgvector(embedding_func)
    if pgvector_db:
        start_time = time.time()
        # Use the same batch size for fair comparison
        for i in range(0, len(documents), 1000):
            end_idx = min(i + 1000, len(documents))
            pgvector_db.add_documents(documents[i:end_idx])
        pgvector_time = time.time() - start_time
        results.append({"database": "PGVector", "operation": "insertion",
                       "dataset_size": dataset_size, "time_seconds": pgvector_time})

    return results

def benchmark_query(dataset_size, num_queries):
    """Benchmark query performance"""
    documents, embedding_func, embeddings_dict = get_test_data(dataset_size)

    # Generate query texts (use some from the dataset)
    query_indices = np.random.choice(dataset_size, num_queries, replace=False)
    query_texts = [documents[i].page_content for i in query_indices]

    results = []

    # ChromaDB setup and insertion
    chroma_db = setup_chroma(embedding_func)
    if chroma_db:
        # Process in batches of 1000 to avoid exceeding ChromaDB's limit
        for i in range(0, len(documents), 1000):
            end_idx = min(i + 1000, len(documents))
            chroma_db.add_documents(documents[i:end_idx])

        # Query benchmark
        start_time = time.time()
        for query_text in query_texts:
            chroma_db.similarity_search(query_text, k=10)
        chroma_time = time.time() - start_time
        results.append({"database": "ChromaDB", "operation": "query",
                       "dataset_size": dataset_size, "num_queries": num_queries,
                       "time_seconds": chroma_time})

    # PGVector setup and insertion
    pgvector_db = setup_pgvector(embedding_func)
    if pgvector_db:
        # Use the same batch size for fair comparison
        for i in range(0, len(documents), 1000):
            end_idx = min(i + 1000, len(documents))
            pgvector_db.add_documents(documents[i:end_idx])

        # Query benchmark
        start_time = time.time()
        for query_text in query_texts:
            pgvector_db.similarity_search(query_text, k=10)
        pgvector_time = time.time() - start_time
        results.append({"database": "PGVector", "operation": "query",
                       "dataset_size": dataset_size, "num_queries": num_queries,
                       "time_seconds": pgvector_time})

    return results

def get_container_memory(container_pattern):
    """Get memory usage of a Docker container matching the pattern"""
    import subprocess  # Add this import statement

    try:
        # List all containers that match the pattern
        cmd = f"docker ps --format '{{{{.Names}}}}' | grep -i {container_pattern}"
        print(f"\nSearching for containers matching pattern: {container_pattern}")
        container_names = subprocess.check_output(cmd, shell=True).decode().strip().split('\n')

        if not container_names or not container_names[0]:
            print(f"No containers found matching pattern: {container_pattern}")
            return 0

        print(f"Found containers: {container_names}")
        container_name = container_names[0]  # Use the first matching container

        # Get container stats
        cmd = f"docker stats {container_name} --no-stream --format '{{{{.MemUsage}}}}'"
        print(f"Getting stats for container: {container_name}")
        stats = subprocess.check_output(cmd, shell=True).decode().strip()
        print(f"Raw stats output: {stats}")

        # Parse memory value (format could be "100MiB / 16GiB" or "1.2GiB / 16GiB")
        memory_part = stats.split('/')[0].strip()

        if 'MiB' in memory_part:
            memory_used = float(memory_part.split('MiB')[0].strip())
        elif 'GiB' in memory_part:
            memory_used = float(memory_part.split('GiB')[0].strip()) * 1024  # Convert GiB to MiB
        else:
            print(f"Unknown memory format: {memory_part}")
            return 0

        print(f"Parsed memory usage: {memory_used} MB")
        return memory_used

    except Exception as e:
        print(f"Error getting container memory: {e}")
        return 0

def benchmark_memory_usage(dataset_size):
    """Benchmark memory usage by measuring Docker container memory"""
    import gc
    import time

    documents, embedding_func, _ = get_test_data(dataset_size)
    results = []

    # ChromaDB memory usage
    print(f"\nMeasuring ChromaDB memory usage for {dataset_size} documents...")
    gc.collect()
    time.sleep(1)

    # Get initial ChromaDB container memory
    initial_memory = get_container_memory("chroma")
    print(f"Initial ChromaDB memory: {initial_memory} MB")

    chroma_db = setup_chroma(embedding_func)
    if chroma_db:
        # Process in batches and measure after each batch
        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            chroma_db.add_documents(documents[i:end_idx])

            # Measure and print progress
            current_memory = get_container_memory("chroma")
            memory_diff = current_memory - initial_memory
            print(f"Progress: {end_idx}/{len(documents)} documents")
            print(f"Current memory: {current_memory:.2f} MB (Difference: {memory_diff:.2f} MB)")
            time.sleep(1)  # Allow memory stats to stabilize

        # Final memory measurement
        time.sleep(2)  # Allow final memory stats to stabilize
        final_memory = get_container_memory("chroma")
        chroma_memory = max(0, final_memory - initial_memory)
        results.append({"database": "ChromaDB", "operation": "memory_usage",
                       "dataset_size": dataset_size, "memory_mb": chroma_memory})

        print(f"ChromaDB final memory usage: {chroma_memory:.2f} MB")

    # Reset ChromaDB
    chroma_db = None
    gc.collect()
    time.sleep(2)

    # PGVector memory usage
    print(f"\nMeasuring PGVector memory usage for {dataset_size} documents...")

    # Get initial PGVector container memory
    initial_memory = get_container_memory("postgres")
    print(f"Initial PGVector memory: {initial_memory} MB")

    pgvector_db = setup_pgvector(embedding_func)
    if pgvector_db:
        # Process in batches and measure after each batch
        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            pgvector_db.add_documents(documents[i:end_idx])

            # Measure and print progress
            current_memory = get_container_memory("postgres")
            memory_diff = current_memory - initial_memory
            print(f"Progress: {end_idx}/{len(documents)} documents")
            print(f"Current memory: {current_memory:.2f} MB (Difference: {memory_diff:.2f} MB)")
            time.sleep(1)  # Allow memory stats to stabilize

        # Final memory measurement
        time.sleep(2)  # Allow final memory stats to stabilize
        final_memory = get_container_memory("postgres")
        pgvector_memory = max(0, final_memory - initial_memory)
        results.append({"database": "PGVector", "operation": "memory_usage",
                       "dataset_size": dataset_size, "memory_mb": pgvector_memory})

        print(f"PGVector final memory usage: {pgvector_memory:.2f} MB")

        # Clean up
        pgvector_db = None
        gc.collect()

    return results

def benchmark_cpu_usage(dataset_size, num_queries):
    """Benchmark CPU usage during queries"""
    documents, embedding_func, _ = get_test_data(dataset_size)

    # Generate query texts
    query_indices = np.random.choice(dataset_size, num_queries, replace=False)
    query_texts = [documents[i].page_content for i in query_indices]

    results = []

    # ChromaDB CPU usage
    chroma_db = setup_chroma(embedding_func)
    if chroma_db:
        # Process in batches of 1000 to avoid exceeding ChromaDB's limit
        for i in range(0, len(documents), 1000):
            end_idx = min(i + 1000, len(documents))
            chroma_db.add_documents(documents[i:end_idx])

        process = psutil.Process(os.getpid())
        start_cpu_times = process.cpu_times()
        start_time = time.time()

        for query_text in query_texts:
            chroma_db.similarity_search(query_text, k=10)

        end_time = time.time()
        end_cpu_times = process.cpu_times()

        cpu_user = end_cpu_times.user - start_cpu_times.user
        cpu_system = end_cpu_times.system - start_cpu_times.system
        elapsed = end_time - start_time

        # Calculate CPU usage percentage
        cpu_percent = (cpu_user + cpu_system) / elapsed * 100

        results.append({"database": "ChromaDB", "operation": "cpu_usage",
                       "dataset_size": dataset_size, "num_queries": num_queries,
                       "cpu_percent": cpu_percent})

    # PGVector CPU usage
    pgvector_db = setup_pgvector(embedding_func)
    if pgvector_db:
        # Use the same batch size for fair comparison
        for i in range(0, len(documents), 1000):
            end_idx = min(i + 1000, len(documents))
            pgvector_db.add_documents(documents[i:end_idx])

        process = psutil.Process(os.getpid())
        start_cpu_times = process.cpu_times()
        start_time = time.time()

        for query_text in query_texts:
            pgvector_db.similarity_search(query_text, k=10)

        end_time = time.time()
        end_cpu_times = process.cpu_times()

        cpu_user = end_cpu_times.user - start_cpu_times.user
        cpu_system = end_cpu_times.system - start_cpu_times.system
        elapsed = end_time - start_time

        # Calculate CPU usage percentage
        cpu_percent = (cpu_user + cpu_system) / elapsed * 100

        results.append({"database": "PGVector", "operation": "cpu_usage",
                       "dataset_size": dataset_size, "num_queries": num_queries,
                       "cpu_percent": cpu_percent})

    return results




def run_all_benchmarks():
    """Run all benchmarks and save results to CSV files"""
    # List all running Docker containers for debugging
    print("\nListing all running Docker containers:")

    # Insertion benchmark
    insertion_results = []
    for size in DATASET_SIZES:
        print(f"Running insertion benchmark with {size} documents...")
        insertion_results.extend(benchmark_insertion(size))

    df = pd.DataFrame(insertion_results)
    df.to_csv("benchmark_results/insertion_benchmark.csv", index=False)

    # Query benchmark
    query_results = []
    for size in DATASET_SIZES:
        print(f"Running query benchmark with {size} documents...")
        query_results.extend(benchmark_query(size, NUM_QUERIES))

    df = pd.DataFrame(query_results)
    df.to_csv("benchmark_results/query_benchmark.csv", index=False)

    # Memory usage benchmark
    memory_results = []
    for size in DATASET_SIZES:
        print(f"Running memory usage benchmark with {size} documents...")
        memory_results.extend(benchmark_memory_usage(size))
        # Print current results after each dataset size
        df_temp = pd.DataFrame(memory_results)
        print("\nCurrent memory usage results:")
        print(df_temp)

    df = pd.DataFrame(memory_results)
    df.to_csv("benchmark_results/memory_usage_benchmark.csv", index=False)

    # # CPU usage benchmark
    cpu_results = []
    for size in DATASET_SIZES:
        print(f"Running CPU usage benchmark with {size} documents...")
        cpu_results.extend(benchmark_cpu_usage(size, NUM_QUERIES))

    df = pd.DataFrame(cpu_results)
    df.to_csv("benchmark_results/cpu_usage_benchmark.csv", index=False)

def generate_charts():
    """Generate charts from benchmark results"""
    os.makedirs("benchmark_results/charts", exist_ok=True)

    # Insertion time chart
    if os.path.exists("benchmark_results/insertion_benchmark.csv"):
        df = pd.read_csv("benchmark_results/insertion_benchmark.csv")
        plt.figure(figsize=(10, 6))
        for db in df['database'].unique():
            data = df[df['database'] == db]
            plt.plot(data['dataset_size'], data['time_seconds'], marker='o', label=db)
        plt.xlabel('Dataset Size')
        plt.ylabel('Time (seconds)')
        plt.title('Insertion Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig("benchmark_results/charts/insertion_performance.png")

    # Query time chart
    if os.path.exists("benchmark_results/query_benchmark.csv"):
        df = pd.read_csv("benchmark_results/query_benchmark.csv")
        plt.figure(figsize=(10, 6))
        for db in df['database'].unique():
            data = df[df['database'] == db]
            plt.plot(data['dataset_size'], data['time_seconds'], marker='o', label=db)
        plt.xlabel('Dataset Size')
        plt.ylabel('Time (seconds)')
        plt.title('Query Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig("benchmark_results/charts/query_performance.png")

    # Memory usage chart
    if os.path.exists("benchmark_results/memory_usage_benchmark.csv"):
        df = pd.read_csv("benchmark_results/memory_usage_benchmark.csv")
        plt.figure(figsize=(10, 6))
        for db in df['database'].unique():
            data = df[df['database'] == db]
            plt.plot(data['dataset_size'], data['memory_mb'], marker='o', label=db)
        plt.xlabel('Dataset Size')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage')
        plt.legend()
        plt.grid(True)
        plt.savefig("benchmark_results/charts/memory_usage.png")

    # CPU usage chart
    if os.path.exists("benchmark_results/cpu_usage_benchmark.csv"):
        df = pd.read_csv("benchmark_results/cpu_usage_benchmark.csv")
        plt.figure(figsize=(10, 6))
        for db in df['database'].unique():
            data = df[df['database'] == db]
            plt.plot(data['dataset_size'], data['cpu_percent'], marker='o', label=db)
        plt.xlabel('Dataset Size')
        plt.ylabel('CPU Usage (%)')
        plt.title('CPU Usage')
        plt.legend()
        plt.grid(True)
        plt.savefig("benchmark_results/charts/cpu_usage.png")

def measure_memory_usage(func, *args, **kwargs):
    """
    Measure memory usage of a function call more accurately

    Args:
        func: Function to measure
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Tuple of (function result, memory usage in MB)
    """
    import gc
    import time

    # Force garbage collection to get a clean slate
    gc.collect()
    time.sleep(0.5)

    # Get initial memory usage
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    # Call the function
    result = func(*args, **kwargs)

    # Force garbage collection again to clean up any temporary objects
    gc.collect()
    time.sleep(0.5)

    # Get final memory usage
    memory_after = process.memory_info().rss / 1024 / 1024  # MB

    # Calculate memory usage (ensure it's not negative)
    memory_used = max(0, memory_after - memory_before)

    return result, memory_used

if __name__ == "__main__":
    print("Starting ChromaDB vs PGVector benchmarks...")
    run_all_benchmarks()
    print("Generating charts from benchmark results...")
    generate_charts()
    print("Benchmarks completed. Results saved in 'benchmark_results' directory.")
