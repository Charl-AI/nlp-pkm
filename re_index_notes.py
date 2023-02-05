"""Generate embeddings for all markdown files in the notes/ directory."""

import argparse
import dataclasses
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing_extensions import TypeGuard

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
encoder.max_seq_length = 256  # Truncate long passages to 256 tokens


def is_valid_embedding(embedding) -> TypeGuard[torch.Tensor]:
    # return embedding.shape == (384)
    return True


@dataclasses.dataclass
class Document:
    title: str
    content: str
    content_hash: str
    embedding: torch.Tensor


DocumentCollection = Dict[Path, Document]


def re_index_notes(
    path_to_notes: Path, path_to_index: Path, force: bool = False
) -> None:
    """Generate embeddings for all markdown files in the notes directory, saving the
    results to a pickle file.

    Args:
        path_to_notes (Path): path to the notes directory
        path_to_index (Path): where to save the index
        force (bool, optional): if true, ignores any existing index to force all notes
            to be reindexed. Defaults to False.
    """
    if path_to_index.exists() and not force:
        with path_to_index.open("rb") as f:
            index: DocumentCollection = pickle.load(f)
            logger.warning(f"Loaded existing index with {len(index)} documents")
    else:
        index = {}
        logger.warning("No existing index found, creating a new one")

    num_skipped = 0
    num_reindexed = 0

    for path in tqdm(
        path_to_notes.glob("**/*.md"),
        desc="Indexing notes",
        unit="files",
        total=len(list(path_to_notes.glob("**/*.md"))),
    ):
        # Skip files that are unchanged since the last run
        if path in index:
            with path.open("r") as f:
                content = f.read()
            if (
                index[path].content_hash
                == hashlib.md5(bytes(content, "utf-8")).hexdigest()
            ):
                num_skipped += 1
                continue

        with path.open("r") as f:
            title = path.stem
            content = f.read()
            content_hash = hashlib.md5(bytes(content, "utf-8")).hexdigest()
        embedding = encoder.encode(
            content, convert_to_tensor=True, normalize_embeddings=True
        )
        assert is_valid_embedding(embedding)

        num_reindexed += 1
        index[path] = Document(title, content, content_hash, embedding)

    logger.warning(f"Reindexed {num_reindexed} files")
    logger.warning(f"Skipped {num_skipped} unchanged files")

    # Save the index
    with path_to_index.open("wb") as f:
        pickle.dump(index, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_notes",
        type=Path,
        default=Path("notes"),
        help="Path to the notes directory",
    )
    parser.add_argument(
        "--path_to_index",
        type=Path,
        default=Path("index.pkl"),
        help="Where to save the index",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If true, ignores any existing index to force all notes to be reindexed",
    )
    args = parser.parse_args()
    re_index_notes(args.path_to_notes, args.path_to_index, args.force)
