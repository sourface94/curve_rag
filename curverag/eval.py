from typing import List

from sentence_transformers import SentenceTransformer
from curverag.curverag import CurveRAG

queries = [
    'How old is Jack?',
    'What gender is Jill?'
    'Give me the character arc of Sam',
    'What location is the story set in',
    'Tell me how old Jack is, what the gender of Jill is, and then give me the character arcs of Sam and where the stoty is set'
    'Give me a summary of the story',
]


expected_output = [
    'Jack is 45 years old',
    'Jill is a woman',
    'Sam starts the story sad and then meets a partner and becomes happy as they live together in a Castle in Romania',
    'Jack is 45 years old, Jill is a woman and Sam starts the story sad. Sams character arc is hen meets a partner and becomes happy as they live together in a Castle in ROmania',
    'The story is about Jack and Jill who are siblings who meet a person called Sam who is very sad but then Jack and Jill introduce Sam to their friend, who Sam starts to date andd they ,pde to Romania where they live happpily ever after.'
]

context = """
Jack and Jill are adult siblings. Jakc just turnt 45 years old as it was his birthday 2 days ago and Jill is also 45; you gussed it they are twins. Jack and Jill bumped into their friend Sam and Sam seemed to be very upset. They know Sam has been single for a long time so they told Sam to talk to their friend who they think could be a good match for Sam. One year later and Sam and hey friend were dating and had even moved to Romania in a massive Castle. Sam is very happy now and they have a "happily ever after" story, which is fantastic.
"""


def evaluation(cg: CurveRAG, context: List[str], queries: List[str], expected_output: List['str']):
    cg.fit([context])
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    expected_output_emb = st_model.encode(expected_output)

    similarities = []
    for i, q in enumerate(queries):
        res = cg.query(q)
        res_emb = st_model.encode(res)
        sim = st_model.similarity([res_emb], [expected_output_emb[i]])
        similarities.append(sim[0])

if __name__ == "__main__":
    max_tokens = 10000
    n_ctx = 
    model = utils.load_model(
        llm_model_path="./models/Meta-Llama-3-8B-Instruct.Q6_K.gguf",
        tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct"),
        n_ctx=n_ctx,
        max_tokens=max_tokens
    )
    

    evaluation([context], queries, expected_output):
    