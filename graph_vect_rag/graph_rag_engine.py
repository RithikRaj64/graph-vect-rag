# Import necessary libraries and modules
import json
import os

import uuid

from dotenv import load_dotenv

from typing import Optional

from llama_index.core import (
    Settings,
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KGTableRetriever, VectorIndexRetriever
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore

# Importing a custom retriever class from another module
from graph_vect_rag.retreiver import CustomRetriever

# Importing exceptions
import graph_vect_rag.exceptions as ex

# LLM Models supported by GROQ
groq_models_list = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
    "gemma2-9b-it",
]

# Default value for data.json file
default_json = {
    "knowledge_bases": [],
    "knowledge_base_id": {},
    "knowledge_base_path": {},
}


# Class to manage and interact with a knowledge base represented as a graph and vector store
class Graph_RAG_Engine:

    # Class attributes
    storage_context: StorageContext = (
        None  # Stores the context for graph and vector storage
    )
    graph_vector_rag_query_engine: RetrieverQueryEngine = (
        None  # Query engine for retrieval and generation
    )
    llm = None  # Language model instance
    embed_model: HuggingFaceEmbedding = None  # Embedding model instance
    context_window: int = 2048  # Maximum context window size
    chunk_size: int = 512  # Size of each data chunk

    # Initialize class and set global settings for LLM, embedding model, chunk size, and context window
    def __init__(self) -> None:
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = self.chunk_size
        Settings.context_window = self.context_window

    # Method to configure the language model (LLM)
    def configure_llm(
        self,
        model_id: str,
        context_window: Optional[int] = None,
        chunk_size: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:

        if model_id not in groq_models_list:
            raise ex.InvlaidModelIdException(
                f"Model_id should be one of the following {groq_models_list}"
            )

        if not api_key:
            load_dotenv()
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key is None:
                raise ValueError("The environment variable 'GROQ_API_KEY' is not set.")

        llm = Groq(
            model=model_id,
            api_key=api_key,
        )

        self.llm = llm
        if context_window:
            self.context_window = context_window
        if chunk_size:
            self.chunk_size = chunk_size
        Settings.llm = llm
        Settings.context_window = self.context_window
        Settings.chunk_size = self.chunk_size

    # Method to configure the embedding model using HuggingFace
    def configure_embedding_model(self, model_id: str) -> None:
        embed_model = HuggingFaceEmbedding(model_name=model_id)
        self.embed_model = embed_model
        Settings.embed_model = embed_model

    # Method to configure the graph store using Neo4j database
    def configure_graph_store(
        self,
        password: str,
        username: Optional[str] = "neo4j",
        url: Optional[str] = "bolt://localhost:7687",
    ) -> None:
        graph_store = Neo4jGraphStore(username=username, password=password, url=url)
        self.storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # Method to create a knowledge graph and vector store from a document
    def create_knowledge_base(self, file_path: str, knowledge_base_name: str) -> None:
        # Create JSON file if it doesnt exist
        if not os.path.exists("data.json"):
            with open("data.json", "w") as json_file:
                json.dump(default_json, json_file)

        # Check if the graph store, LLM and embedding model have been configured
        if not self.storage_context:
            raise ex.StorageContextNotFoundException("Configure your Graph Store")

        if not self.llm:
            raise ex.LLMNotFoundException("Configure your LLM")

        if not self.embed_model:
            raise ex.EmbeddingModelNotFoundException("Configure your Embedding Model")

        # Load JSON data
        with open("data.json", "r") as jsonFile:
            data = json.load(jsonFile)

        knowledge_bases = data["knowledge_bases"]

        if knowledge_base_name in data["knowledge_bases"]:
            raise ex.KnowledgeBaseAlreadyExists(
                f"Knowledge Base name should be unique. Already existing KBs {knowledge_bases}"
            )

        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()  # Read documents

        # Create KnowledgeGraphIndex and VectorStoreIndex from the documents
        kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            max_triplets_per_chunk=10,
            include_embeddings=True,
            show_progress=True,
        )

        vector_index = VectorStoreIndex.from_documents(documents)

        unique_id = str(uuid.uuid4())

        kg_index.set_index_id(f"{unique_id}_graph_index")
        vector_index.set_index_id(f"{unique_id}_vector_index")

        # Persist the indexes to disk
        kg_index.storage_context.persist(
            persist_dir=f"./storage/{knowledge_base_name}_{unique_id}/graph"
        )
        vector_index.storage_context.persist(
            persist_dir=f"./storage/{knowledge_base_name}_{unique_id}/vector"
        )

        # Update JSON data file to include the new knowledge base path
        data["knowledge_bases"].append(knowledge_base_name)

        data["knowledge_base_path"].update(
            {knowledge_base_name: f"./storage/{knowledge_base_name}_{unique_id}"}
        )

        data["knowledge_base_id"].update({knowledge_base_name: unique_id})

        with open("data.json", "w") as jsonFile:
            json.dump(data, jsonFile)

    # Method to load a previously created knowledge graph and vector store
    def load_knowledge_base(self, knowledge_base_name: str) -> bool:
        # Create JSON file if it doesnt exist
        if not os.path.exists("data.json"):
            with open("data.json", "w") as json_file:
                json.dump(default_json, json_file)

        with open("data.json", "r") as jsonFile:
            data = json.load(jsonFile)

        knowledge_bases = data["knowledge_bases"]

        if knowledge_base_name not in knowledge_bases:
            raise ex.KnowledgeBaseNotFound(
                f"Invalid Knowledge Base name. Existing KBs {knowledge_bases}"
            )

        # Extract information to access Knowledge Base
        kb_path = data["knowledge_base_path"].get(knowledge_base_name)
        unique_id = data["knowledge_base_id"].get(knowledge_base_name)

        # Load graph and vector storage contexts from disk
        graph_context = StorageContext.from_defaults(persist_dir=f"{kb_path}/graph")
        vector_context = StorageContext.from_defaults(persist_dir=f"{kb_path}/vector")

        # Load the knowledge graph and vector index from storage
        graph_index = load_index_from_storage(
            storage_context=graph_context,
            index_id=f"{unique_id}_graph_index",
            max_triplets_per_chunk=10,
            include_embeddings=True,
            show_progress=True,
        )

        vector_index = load_index_from_storage(
            storage_context=vector_context, index_id=f"{unique_id}_vector_index"
        )

        # Create retrievers for both the graph and vector indexes
        graph_retriever = KGTableRetriever(
            index=graph_index,
            retriever_mode="keyword",
            include_text=False,
            verbose=True,
        )
        vector_retriever = VectorIndexRetriever(index=vector_index, verbose=True)

        # Combine the two retrievers into a custom retriever
        custom_retriever = CustomRetriever(
            vector_retriever=vector_retriever, kg_retriever=graph_retriever
        )

        # Create a response synthesizer to aggregate and process responses
        response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

        # Instantiate the query engine that combines both retrievers and the response synthesizer
        self.graph_vector_rag_query_engine = RetrieverQueryEngine(
            retriever=custom_retriever, response_synthesizer=response_synthesizer
        )

        return True

    # Method to query the knowledge graph using natural language queries
    def query_knowledge_base(self, query: str) -> str:
        # Check if the Knowledge Base has been selected
        if self.graph_vector_rag_query_engine is None:
            raise ex.KnowledgeBaseNotConfigured(
                "Select your KB using load_knowledge_base function before querying."
            )

        return self.graph_vector_rag_query_engine.query(query).response
