# langchain_pgvector_searchkit
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
'''
