from typing import Sequence, Any
from sqlalchemy.engine import RowMapping


def hybrid_search_rrf_weighted_sum(
    primary_search_results: Sequence[RowMapping],
    secondary_search_results: Sequence[RowMapping],
    primary_rrf_k: float = 60,
    secondary_rrf_k: float = 60,
    primary_weight: float = 0.5,
    secondary_weight: float = 0.5,
    fetch_top_k: int = 4,
) -> Sequence[dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}

    for rank, row in enumerate(primary_search_results):
        doc_id = str(row["id"])
        rrf_score = 1.0 / (rank + primary_rrf_k) * primary_weight
        prev_score = scores.get(doc_id, {}).get("score", 0.0)

        row_map = dict(row)
        row_map["score"] = prev_score + rrf_score  
        scores[doc_id] = row_map
         
        
    for rank, row in enumerate(secondary_search_results):
        doc_id = str(row["id"])
        rrf_score = 1.0 / (rank + secondary_rrf_k) * secondary_weight
        prev_score = scores.get(doc_id, {}).get("score", 0.0)

        row_map = dict(row)
        row_map["score"] = prev_score + rrf_score 
        scores[doc_id] = row_map
    
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)[:fetch_top_k]

    return ranked
