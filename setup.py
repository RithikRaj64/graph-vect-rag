from setuptools import setup, find_packages

setup(
    name="graph_vect_rag",
    version="1.0.0",
    author="Rithik Raj K S",
    author_email="rithu0644@gmail.com",
    packages=find_packages(),
    install_requires=[
        "llama-index",
        "llama-index-llms-groq",
        "llama-index-embeddings-huggingface",
        "llama-index-graph-stores-neo4j",
        "sentence-transformers",
        "python-dotenv",
        "setuptools",
    ],
    description="A Python package for hybrid Graph + Vector RAG using completely Open Source tools",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    project_urls={
        "Source": "https://github.com/RithikRaj64/graph-vect-rag",
        "Bug Tracker": "https://github.com/RithikRaj64/graph-vect-rag/issues",
    },
)
