# Graph_Vect_RAG

`graph_vect_rag` is a powerful Python package designed to manage and query knowledge bases using a combination of knowledge graph and vector stores. This package supports integration with GROQ language models, HuggingFace embeddings, and Neo4j graph stores to build and query hybrid retrieval-augmented generation (RAG) systems.

## Features

- **Language Model (LLM) Integration**: Supports multiple language models from GROQ for natural language processing.
- **Graph and Vector Store Management**: Handles knowledge bases as a combination of graph and vector indexes.
- **Custom Retriever**: Combines graph and vector store retrievers for more effective query results.
- **Embeddings with HuggingFace**: Uses HuggingFace embeddings for vector indexing.
- **Neo4j Graph Store Integration**: Manages graph stores using the Neo4j database.
- **Flexible Querying**: Allows natural language queries to the knowledge base.

## Installation

To install `Graph_Vect_RAG`, use pip:

```bash
pip install graph-vect-rag
```

## Basic Setup

### 1. **Initialize the Engine**:

```python
from graph_rag_engine import Graph_RAG_Engine

engine = Graph_RAG_Engine()
```

### 2. **Configure the Language Model (LLM)**:

You can configure the language model using any model from the supported list provided at the end of the document:

- **Directly providing the API key**:

  ```python
  engine.configure_llm(
      model_id="model_name_from_supported_list",
      api_key="your_groq_api_key"
  )
  ```

- **Using an environment variable**:
  Alternatively, you can set the API key in your environment by creating a `.env` file with the following content:
  ```env
  GROQ_API_KEY=your_groq_api_key
  ```
  The package will automatically load the API key from the environment variable `GROQ_API_KEY` when the `api_key` argument is not provided.

Replace `"model_name_from_supported_list"` with any of the supported models listed at the end of the document, such as `"llama-3.1-70b-versatile"` or `"llama-3.1-8b-instant"`.

### 3. **Configure the Embedding Model**:

You can configure the embedding model using any HuggingFace embedding model:

```python
engine.configure_embedding_model(model_id="huggingface_model_name")
```

Replace `"huggingface_model_name"` with the name of any HuggingFace model you wish to use for embeddings (e.g., `"sentence-transformers/all-MiniLM-L6-v2"`).

### 4. **Configure the Graph Store**:

You can configure the graph store with default values for `username` and `url`. The default values are `username="neo4j"` and `url="bolt://localhost:7687"`:

```python
engine.configure_graph_store(
    password="neo4j_password",
    username="neo4j",  # Default value
    url="bolt://localhost:7687"  # Default value
)
```

If you need to use different values, simply provide them as arguments to `configure_graph_store`.

> **Note**: You need to have Neo4j installed and running. Additionally, make sure that APOC procedures are enabled in your Neo4j instance for full functionality with this package. You can download Neo4j from its official website: [Neo4j Download Page](https://neo4j.com/download/) For detailed installation instructions and additional resources, visit the Neo4j documentation page: [Neo4j Documentation](https://neo4j.com/docs/)

## Create a Knowledge Base

To create a knowledge base from a document:

```python
engine.create_knowledge_base(
    file_path="path/to/your/document.txt",
    knowledge_base_name="my_knowledge_base"
)
```

## Load a Knowledge Base

To load an existing knowledge base:

```python
engine.load_knowledge_base(knowledge_base_name="my_knowledge_base")
```

## Query the Knowledge Base

To query the loaded knowledge base using natural language:

```python
response = engine.query_knowledge_base("What is the capital of France?")
print(response)
```

## Exception Handling

`Graph_Vect_RAG` provides custom exceptions to handle various error scenarios:

- `InvlaidModelIdException`: Raised when an invalid model ID is provided.
- `StorageContextNotFoundException`: Raised when the storage context is not configured.
- `LLMNotFoundException`: Raised when the LLM is not configured.
- `EmbeddingModelNotFoundException`: Raised when the embedding model is not configured.
- `KnowledgeBaseAlreadyExists`: Raised when trying to create a knowledge base with an existing name.
- `KnowledgeBaseNotFound`: Raised when attempting to load a non-existent knowledge base.
- `KnowledgeBaseNotConfigured`: Raised when querying without selecting a knowledge base.

## Supported GROQ Models

The following models are supported by the package:

- `llama-3.1-70b-versatile`
- `llama-3.1-8b-instant`
- `llama3-groq-70b-8192-tool-use-preview`
- `llama3-groq-8b-8192-tool-use-preview`
- `llama-guard-3-8b`
- `llama3-70b-8192`
- `llama3-8b-8192`
- `mixtral-8x7b-32768`
- `gemma-7b-it`
- `gemma2-9b-it`
