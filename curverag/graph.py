from typing import List


from curverag.transformations import chunk_text
from curverag.prompts import entity_relationship_extraction__disparate_prompt


def create_graph(texts: List[str], is_narrative: bool = False):
    """
    Create knowledge graph. 

    Creation of this packages knowledge center 
    """
    
    # chunk text
    texts = chunk_text(texts)





