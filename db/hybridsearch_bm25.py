from __future__ import annotations
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from typing import Any, Optional, Sequence
from sqlalchemy import RowMapping, text
from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig
import re
class AsyncPGVectorStoreBM25(AsyncPGVectorStore):
    async def _AsyncPGVectorStore__query_collection(
        self,
        embedding: list[float],
        *,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
        ) -> Sequence[RowMapping] :
        
        if not k:
            k = (
                max(
                    self.k,
                    self.hybrid_search_config.primary_top_k,
                    self.hybrid_search_config.secondary_top_k,
                )
                if self.hybrid_search_config
                else self.k
            )
        operator = self.distance_strategy.operator
        search_function = self.distance_strategy.search_function

        columns = [
            self.id_column,
            self.content_column,
            self.embedding_column,
        ] + self.metadata_columns
        if self.metadata_json_column:
            columns.append(self.metadata_json_column)

        column_names = ", ".join(f'"{col}"' for col in columns)

        safe_filter = None
        filter_dict = None
        if filter and isinstance(filter, dict):
            safe_filter, filter_dict = self._create_filter_clause(filter)
        inline_embed_func = getattr(self.embedding_service, "embed_query_inline", None)
        if not embedding and callable(inline_embed_func) and "query" in kwargs:
            query_embedding = self.embedding_service.embed_query_inline(kwargs["query"])
            embedding_data_string = f"{query_embedding}"
        else:
            
            query_embedding = f"{[float(dimension) for dimension in embedding]}"
            embedding_data_string = "(:query_embedding)::vector"
            
        where_filters = f"WHERE {safe_filter}" if safe_filter else ""
        
        if self.hybrid_search_config and self.hybrid_search_config.primary_top_k :
            vector_search_k=self.hybrid_search_config.primary_top_k
        else :
            vector_search_k=k
        dense_query_stmt = f"""SELECT {column_names}, {search_function}("{self.embedding_column}", {embedding_data_string}) as distance
        FROM "{self.schema_name}"."{self.table_name}" {where_filters} ORDER BY "{self.embedding_column}" {operator} {embedding_data_string} LIMIT :k;
        """
        param_dict = {"query_embedding": query_embedding, "k": vector_search_k}

        if filter_dict:
            param_dict.update(filter_dict)
        if self.index_query_options:
            async with self.engine.connect() as conn:
                # Set each query option individually
                for query_option in self.index_query_options.to_parameter():
                    query_options_stmt = f"SET LOCAL {query_option};"
                    await conn.execute(text(query_options_stmt))
                print(dense_query_stmt, param_dict)
                result = await conn.execute(text(dense_query_stmt), param_dict)
                result_map = result.mappings()
                dense_results = result_map.fetchall()
        else:
            async with self.engine.connect() as conn:
                result = await conn.execute(text(dense_query_stmt), param_dict)
                result_map = result.mappings()
                dense_results = result_map.fetchall()

        if self.hybrid_search_config : 
            hybrid_search_config = kwargs.get(
                "hybrid_search_config", self.hybrid_search_config
            )

            if hybrid_search_config.index_type == 'BM25' :
                and_filters = f"AND ({safe_filter})" if safe_filter else ""
                fts_query = kwargs.get("query", "")
                sparse_query_stmt= f"""
                    SELECT {column_names}, paradedb.score(id) as distance
                    FROM "{self.schema_name}"."{self.table_name}"
                    WHERE content @@@ paradedb.match('content', :fts_query) {and_filters}
                    ORDER BY distance desc
                    LIMIT {hybrid_search_config.secondary_top_k};
                    """
                param_dict = {"fts_query": fts_query}
                if filter_dict:
                    param_dict.update(filter_dict)
      

                async with self.engine.connect() as conn:
                    result = await conn.execute(text(sparse_query_stmt), param_dict)
                    result_map = result.mappings()
                    sparse_results = result_map.fetchall()

                hybrid_search_config.fusion_function_parameters['fetch_top_k']=k
                combined_results = hybrid_search_config.fusion_function(
                    dense_results,
                    sparse_results,
                    **hybrid_search_config.fusion_function_parameters,
                )
                return combined_results
        else : 
            return dense_results

        # When need basic pgvector's Full text search 
        # if self.hybrid_search_config : 
        #     hybrid_search_config = kwargs.get(
        #         "hybrid_search_config", self.hybrid_search_config
        #     )

        #     if hybrid_search_config.index_type == 'basic' :
                # fts_query = (
                #     hybrid_search_config.fts_query
                #     if hybrid_search_config and hybrid_search_config.fts_query
                #     else kwargs.get("fts_query", "")
                # )
                # if hybrid_search_config and fts_query:
                #     hybrid_search_config.fusion_function_parameters["fetch_top_k"] = k
                
                #     lang = (
                #         f"'{hybrid_search_config.tsv_lang}',"
                #         if hybrid_search_config.tsv_lang
                #         else ""
                #     )
                #     query_tsv = f"plainto_tsquery({lang} :fts_query)"
                #     param_dict["fts_query"] = fts_query
                #     if hybrid_search_config.tsv_column:
                #         content_tsv = f'"{hybrid_search_config.tsv_column}"'
                #     else:
                #         content_tsv = f'to_tsvector({lang} "{self.content_column}")'
                #     and_filters = f"AND ({safe_filter})" if safe_filter else ""
                #     sparse_query_stmt = f'SELECT {column_names}, ts_rank_cd({content_tsv}, {query_tsv}) as distance FROM "{self.schema_name}"."{self.table_name}" WHERE {content_tsv} @@ {query_tsv} {and_filters}  ORDER BY distance desc LIMIT {hybrid_search_config.secondary_top_k};'
                #     async with self.engine.connect() as conn:
                #         result = await conn.execute(text(sparse_query_stmt), param_dict)
                #         result_map = result.mappings()
                #         sparse_results = result_map.fetchall()

                #     combined_results = hybrid_search_config.fusion_function(
                #         dense_results,
                #         sparse_results,
                #         **hybrid_search_config.fusion_function_parameters,
                #     )
                #     return combined_results


    @classmethod
    def create_sync(cls, **kwargs):
        import asyncio
        return asyncio.run(cls.create(**kwargs))