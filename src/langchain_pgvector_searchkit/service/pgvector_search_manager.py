from langchain_postgres.v2.indexes import HNSWIndex, DistanceStrategy
from langchain_postgres.v2.engine import PGEngine
from sqlalchemy import text
from langchain_huggingface import HuggingFaceEmbeddings
import subprocess
import os
import signal
from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig
from langchain_pgvector_searchkit.rrf.rrf_weighted_sum import hybrid_search_rrf_weighted_sum
import math
from langchain_pgvector_searchkit.db.hybridsearch_bm25 import AsyncPGVectorStoreBM25
import asyncpg
import signal
from typing import Dict
from langchain_core.embeddings import Embeddings
from typing import Self
class PGVectorController :
    def __init__(self
        ,embedding_model_name : str ="nlpai-lab/KURE-v1"
        , model_kwargs : Dict = {"device": "cpu"}
        , encode_kwargs : Dict ={"normalize_embeddings": True}
        , model_type : str = "HuggingFaceEmbeddings") :
        self.embedding_model=self.set_embedding_model(
                 model_name=embedding_model_name
                , model_kwargs=model_kwargs
                , encode_kwargs=encode_kwargs
                , model_type = model_type
            )
        self.connection=None    
        self.connection_url=None
        self.aconnection_url=None
        self.vector_store=None
        self.postgres_server=None
        self.hybrid_cfg=None
        self.retrieval=None

    def set_embedding_model(self
        , model_name : str
        , model_kwargs : Dict = {"device": "cpu"}
        , encode_kwargs : Dict = {"normalize_embeddings": True}
        , model_type : str =  "HuggingFaceEmbeddings") -> Embeddings :
        if model_type=="HuggingFaceEmbeddings":
            return HuggingFaceEmbeddings(
                 model_name=model_name
                , model_kwargs=model_kwargs
                , encode_kwargs=encode_kwargs
            )

    def serve_postgres(self,
        postgresDB_path : str 
        , data_directory : str
        , port : str
        , socket_directory : str
        ) -> Self:
            self.postgres_server = subprocess.Popen([
                    postgresDB_path ,
                    "-D", data_directory ,
                    "-p", port ,
                    "-k", socket_directory
                ],preexec_fn=os.setsid)
            return self

    
    async def aconnect_to_postgres_sql_db(self
        , host: str 
        , port: int 
        , user: str 
        , password: str 
        , dbname: str) -> Self:
        
        self.connection = await asyncpg.connect(
            host=host
            , port=port
            , user=user
            , password=password
            , database=dbname
        )
        self.connection_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.aconnection_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{dbname}"
        version = await self.connection.fetchval("SELECT version();")
        print(version)
        return self

    async def ashow_active_connections(self):
        if self.connection is None:
            print("No DB connection.")
            return
        try:
            sql = """
                SELECT pid, usename, application_name, client_addr, state, query, xact_start
                FROM pg_stat_activity
                WHERE datname = current_database()
                ORDER BY state, xact_start NULLS LAST;
            """
            rows = await self.connection.fetch(sql)
            if not rows:
                print("No active connections.")
                return
            print("Current active connections/transactions status:")
            for row in rows:
                print(f"PID:{row['pid']} USER:{row['usename']} APP:{row['application_name']} IP:{row['client_addr']} state:{row['state']} xact_start:{row['xact_start']}\n  query:{row['query']}\n")
        except Exception as e:
            print(f"Error occurred while retrieving connections: {e}")
    async def akill_all_connections(self):
        if self.connection is None:
            print("No DB connection.")
            return
        try:
            sql_kill = """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = current_database()
                AND pid <> pg_backend_pid();
            """
            await self.connection.execute(sql_kill)

            sql_check = """
                SELECT count(*) AS alive
                FROM pg_stat_activity
                WHERE datname = current_database()
                AND pid <> pg_backend_pid();
            """
            row = await self.connection.fetchrow(sql_check)
            alive = row['alive'] if row else None

            if alive == 0:
                print("All DB sessions have been successfully terminated.")
            else:
                print(f"{alive} sessions are still alive. (Some might have failed to terminate)")
        except Exception as e:
            print("Error occurred while forcibly terminating connections:", e)


    async def astop_postgres_server(self):
        if self.postgres_server is None:
            print("No server process. serve_postgres() must be called first.")
            return

        print(f"Trying to close PostgreSQL... PID: {self.postgres_server.pid}")
        print("\n▶ Active DB connections and transactions before shutdown:")
        await self.ashow_active_connections()
        self.postgres_server.terminate()
        try:
            self.postgres_server.wait(timeout=15)
            print("PostgreSQL closed successfully.")
            self.postgres_server = None
        except subprocess.TimeoutExpired:
            print("Server didn't close normally within timeout.")
            await self.ashow_active_connections()
            user_input = input("Would you like to forcefully terminate all connections and try again? (Y/N): ").strip().lower()
            if user_input == 'y':
                print("Forcefully terminating all connections.")
                await self.akill_all_connections()
                try:
                    self.postgres_server.terminate()
                    self.postgres_server.wait(timeout=15)
                    print("After forcefully terminating connections, PostgreSQL closed successfully.")
                    self.postgres_server = None
                except subprocess.TimeoutExpired:
                    print("The server still did not close.")
                    user_input2 = input("Would you like to force kill the server process with SIGKILL? (Y/N): ").strip().lower()
                    if user_input2 == 'y':
                        os.killpg(os.getpgid(self.postgres_server.pid), signal.SIGKILL)
                        self.postgres_server.wait() 
                        print("PostgreSQL server killed.")
                        self.postgres_server = None
                        print("PostgreSQL server killed and zombie cleaned up.")
                    else:
                        print("Final forced termination was canceled.")
            else:
                print("Forced connection termination and server shutdown were both canceled.")
        

    async def aset_pgvector(
        self,
        table_name: str = "",
        metadata_columns: list[str] = []
        , hybridsearch = True
        , vector_rrf_k : int = 20
        , bm25_rrf_k : int = 60
        , vector_distance_weight : int = 0.5
        , bm25_distance_weight : int = 0.5
        , top_k : int = 5
        , vector_store_cls = AsyncPGVectorStoreBM25
    ) -> AsyncPGVectorStoreBM25 :
        try:
            engine = PGEngine.from_connection_string(self.aconnection_url)
            print("PGEngine initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PGEngine: {e}")     
        try:
            if hybridsearch :
                if not math.isclose(vector_distance_weight + bm25_distance_weight, 1.0, rel_tol=1e-9):
                    raise ValueError("vector_distance_weight + bm25_distance_weight must be 1")
                self.hybrid_cfg = HybridSearchConfig(
                tsv_column= None,
                tsv_lang="simple",
                fusion_function=hybrid_search_rrf_weighted_sum,
                fusion_function_parameters={
                    "primary_rrf_k": vector_rrf_k,        
                    "secondary_rrf_k": bm25_rrf_k,      
                    "primary_weight": vector_distance_weight,      
                    "secondary_weight": bm25_distance_weight,   
                    "fetch_top_k": top_k,
                },
                primary_top_k=100,
                secondary_top_k=100,
                index_name= "bm25_idx",
                index_type= "BM25" # or basic
            )
                self.vector_store = await vector_store_cls.create(
                    engine=engine,
                    embedding_service=self.embedding_model, 
                    table_name=table_name,
                    id_column="id",
                    metadata_columns=metadata_columns,
                    hybrid_search_config=self.hybrid_cfg
                )
            else : 
                self.vector_store = await vector_store_cls.create(
                        engine=engine,
                        embedding_service=self.embedding_model, 
                        table_name=table_name,
                        id_column="id",
                        metadata_columns=metadata_columns
                    )
        except Exception as e:
                raise RuntimeError(f"Failed to initialize PGVectorStore: {e}")

        print(f"PGVectorStore initialized with table: {table_name}")
        return self.vector_store
    

    async def acreate_pgvector_table(self
    , table_name: str = "langchain_pg_embedding"
    , vector_dimension: int = 1024
    , metadata_columns: list[str] = []
    ) -> None:
        columns_sql = ""
        for col in metadata_columns:
            columns_sql += f", {col} TEXT" 

        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id UUID PRIMARY KEY,
            embedding vector({vector_dimension}),
            content TEXT,
            content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
            langchain_metadata JSONB
            {columns_sql}
        );
        """
        try:
            async with self.connection.transaction():
                await self.connection.execute(create_sql)
            print(f"Table '{table_name}' has been created successfully.")
        except Exception as e:
            print(f"Error occurred while creating the table: {e}")


    async def adrop_pgvector_table(self
        , table_name: str = "langchain_pg_embedding"
    ) -> None:
        if self.connection is None:
            print("PostgreSQL connection is not established. Please call aconnect_to_postgres_sql_db() first.")
            return
        try:
            drop_sql = f'DROP TABLE IF EXISTS {table_name} CASCADE;'
            await self.connection.execute(drop_sql)
            print(f"Table '{table_name}' has been dropped.")
        except Exception as e:
            print(f"Failed to drop table '{table_name}': {e}")


    async def adelete_all_docs(self
        , table_name: str = "langchain_pg_embedding"
    ) -> None:
        if self.connection is None:
            print("PostgreSQL connection is not established. Please call aconnect_to_postgres_sql_db() first.")
            return
        try:
            delete_sql = f"DELETE FROM {table_name};"
            await self.connection.execute(delete_sql)
            print(f"All records in the '{table_name}' documents have been deleted.")
        except Exception as e:
            print(f"Error while deleting records via SQL: {e}")
    
    async def acreate_hnsw_index(self
        , table_name : str = None
        , index_name: str = None
        , calc_distance: str = "vector_cosine_ops"  # or vector_ip_ops, vector_l2_ops
        , m: int = 16
        , ef_construction: int = 100,
    ) -> None:
        if calc_distance not in ["vector_ip_ops", "vector_cosine_ops", "vector_l2_ops"]:
            raise ValueError("Invalid calc_distance operator")
        index_name=table_name+"_hnsw_index"
        ops_map = {
            "vector_ip_ops": DistanceStrategy.INNER_PRODUCT,
            "vector_cosine_ops": DistanceStrategy.COSINE_DISTANCE,
            "vector_l2_ops": DistanceStrategy.EUCLIDEAN,
        }
        temp_distance = ops_map[calc_distance]

        await self.vector_store.aapply_vector_index(
        index=HNSWIndex(
            distance_strategy=temp_distance  
            , m=m
            , ef_construction=ef_construction
        ),
        name=index_name, 
        )
        print(f"HNSW index '{index_name}' created successfully ({calc_distance})")

    async def adrop_hnsw_index(self
        , table_name: str
    ) -> None:
        if self.connection is None:
            print("PostgreSQL connection is not established. Please call aconnect_to_postgres_sql_db() first.")
            return
        index_name = f"{table_name}_hnsw_index"
        drop_sql = f'DROP INDEX IF EXISTS {index_name};'
        try:
            await self.connection.execute(drop_sql)
            print(f"HNSW index '{index_name}' deleted successfully")
        except Exception as e:
            print(f"Error occurred while deleting HNSW index: {e}")

    
    async def acreate_bm25_index(self
        , table_name: str
    ) -> None:
        if self.connection is None:
            print("PostgreSQL connection is not established. Please call aconnect_to_postgres_sql_db() first.")
            return
        try:
            index_name = f'bm25_idx_{table_name}'
            create_index_sql = f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {table_name}
                USING bm25 (id, content)
                WITH (
                    key_field = 'id'
                );
            """
            await self.connection.execute(create_index_sql)
            
            check_sql = """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = $1 AND indexname = $2;
            """
            result = await self.connection.fetchrow(check_sql, table_name, index_name)
            if result and 'USING bm25' in result['indexdef']:
                print(f"BM25 index has been successfully created: {result['indexname']}")
            else:
                print("BM25 index was not created or a different type of index exists.")
                print("▶ indexdef:", result['indexdef'] if result else "None")
        except Exception as e:
            print("An error occurred while creating the index:", e)


    async def adrop_bm25_index(self
        , table_name: str
    ) -> None :
        if self.connection is None:
            print("PostgreSQL connection is not established. Please call aconnect_to_postgres_sql_db() first.")
            return
        index_name = f"bm25_idx_{table_name}"
        drop_sql = f"DROP INDEX IF EXISTS {index_name};"
        try:
            await self.connection.execute(drop_sql)
            print(f"BM25 index '{index_name}' deleted successfully")
        except Exception as e:
            print(f"Error occurred while deleting BM25 index: {e}")


    async def acount_docs(self
        , schema: str = "public"
        , table_name: str = None
    ) -> int:
        table = table_name or self.vector_store.table_name
        full_table = f'{schema}.{table}'             
        async with self.vector_store.engine.connect() as conn: 
            try:
                count = await conn.scalar(text(f"SELECT COUNT(*) FROM {full_table};"))
                print(f"total number of docs: {count}")
                return count
            except Exception as e:
                print(f"Error occurred while counting documents: {e}")
                return -1



