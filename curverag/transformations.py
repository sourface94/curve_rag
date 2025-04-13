from typing import List

import semchunk

def chunk_text(texts: List[str], chunk_size: int):
    """
    Chunk the texts

    Many ways to skin a cat
    """

    chunker = semchunk.chunkerify('gpt-4', chunk_size)
    chunks = chunker(texts)

    # NOTE: here we put all texts into a 1d list, which means we arent retaining information within dfocuments
    texts = [j for sub in chunks for j in sub]
    return texts

