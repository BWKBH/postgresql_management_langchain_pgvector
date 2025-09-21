# postgresql_management_langchain_pgvector
Hybrid BM25-powered search &amp; DB management toolkit for PostgreSQL/pgvector with LangChain integration


## install

```bash
git clone https://github.com/BWKBH/langchain_pgvector_searchkit.git
cd langchain_pgvector_searchkit
pip install -e .
pip install -r requirements.txt
```
## Install Required PostgreSQL Extensions

This project depends on the following PostgreSQL extensions:

- [**pgvector**](https://github.com/pgvector/pgvector): for vector similarity search in PostgreSQL  
- [**pg_search (Casecommons)**](https://github.com/Casecommons/pg_search): full-text search functionality using PostgreSQL text search features  
---

### Create Extensions in Your PostgreSQL Database

After installing the extensions, connect to your PostgreSQL instance and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;
```



## Features
- Seamless integration with PostgreSQL + pgvector
- Hybrid search support with BM25 + vector (Reciprocal Rank Fusion)
- Support for HNSW indexing and distance strategy configuration
- Easy control over DB schema, index creation, and deletion
- LangChain-compatible retriever for use in RAG pipelines 

## Quick Usage Example

```python
from langchain_pgvector_searchkit import PGVectorController

controller = PGVectorController(...)
await controller.set_pgvector(...)
retriever = controller.vector_store.as_retriever()
docs = await retriever.ainvoke("한국어 문법에 대해 알려줘", k=4)
```
See more usage in [`test_pgvector_controller.ipynb`](https://github.com/BWKBH/langchain_pgvector_searchkit/blob/main/test_pgvector_controller.ipynb).

## Project Structure
```text
langchain_pgvector_searchkit/
│
├── service/         # PGVectorController and DB management
├── db/              # Hybrid BM25 logic and vector store overrides
└── rrf/             # RRF ranking functions
```

