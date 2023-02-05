import time
import argparse
import logging
import pickle
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer, util, CrossEncoder

from re_index_notes import Document, DocumentCollection, is_valid_embedding

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
bi_encoder.max_seq_length = 256  # Truncate long passages to 256 tokens
top_k = 32  # Number of passages we want to retrieve with the bi-encoder

cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2")


def search(
    path_to_index: Path, query: str, n_results: int, cross_encoder: bool
) -> None:
    if query is None:
        query = input("Enter a query: ")

    with path_to_index.open("rb") as f:
        index: DocumentCollection = pickle.load(f)
        logger.warning(f"Loaded existing index with {len(index)} documents")

    start_time = time.time()
    query_embedding = bi_encoder.encode(
        query, convert_to_tensor=True, normalize_embeddings=True
    )
    assert is_valid_embedding(query_embedding)

    hits = util.semantic_search(
        query_embedding,
        torch.stack([doc.embedding for doc in index.values()], dim=0),
        top_k=top_k if cross_encoder else n_results,
        score_function=util.dot_score,
    )
    hits = hits[0]  # Get the hits for the first (and only) query

    if cross_encoder:
        cross_inp = [
            [query, list(index.values())[hit["corpus_id"]].content] for hit in hits
        ]
        scores = cross_encoder_model.predict(cross_inp)
        # rerank the hits using the cross-encoder scores
        hits = [
            {"corpus_id": hit["corpus_id"], "score": score}
            for hit, score in zip(hits, scores)
        ]
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)

    hits = hits[:n_results]
    docs = [list(index.values())[hit["corpus_id"]] for hit in hits]
    paths = [list(index.keys())[hit["corpus_id"]] for hit in hits]
    print(f"{len(docs)} results in {time.time() - start_time} seconds")
    for doc, path, hit in zip(docs, paths, hits):
        print(f"Score: {hit['score']:.3f} | Title: {doc.title} | Path: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_index",
        type=Path,
        default=Path("index.pkl"),
        help="Path to the index file",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query to search for",
    )
    parser.add_argument(
        "--n_results",
        type=int,
        default=5,
        help="Number of results to return",
    )
    parser.add_argument(
        "--cross_encoder",
        type=bool,
        default=True,
        help="Whether to use second cross encoder stage to improve rankings",
    )
    args = parser.parse_args()
    search(**vars(args))
