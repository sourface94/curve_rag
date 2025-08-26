import re
import json
import string
from collections import defaultdict
from typing import List


from sentence_transformers import SentenceTransformer
from curverag.curverag import CurveRAG

queries = [
    'How old is Jack?',
    'What gender is Jill?',
    'Give me the character arc of Sam',
    'What location is the story set in',
    'Tell me how old Jack is, what the gender of Jill is, and then give me the character arcs of Sam and where the story is set',
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

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)
    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_em_f1(prediction: str, gold_answers: list[str]) -> tuple[float, float]:
    """
    prediction: model output string
    gold_answers: list of possible correct strings (aliases)
    Returns: (EM, F1)
    """
    em = max(compute_em(prediction, ga) for ga in gold_answers)
    f1 = max(compute_f1(prediction, ga) for ga in gold_answers)
    return em, f1

def evaluation(
        model: CurveRAG,
        dataset_path: str = '../datasets/2WikiMultihopQA/new/dev.json',
        dataset_size: int = 1000,
        alias_path: str = '../datasets/2WikiMultihopQA/new/id_aliases.json',
        model_traversal: str = 'pp',
        query_prompt='generate_response_query'
    ):
    """
    Evaluates a CurveRAG model
    """
    print('query_prompt', query_prompt)
    with open(dataset_path, 'rb') as f:
        eval_dataset = json.load(f)
        eval_dataset = eval_dataset[:dataset_size]

    aliases = defaultdict(list)
    with open(alias_path, 'rb') as f:
        for line in f:
            record = json.loads(line)
            aliases[record['Q_id']] += record['aliases'] + record['demonyms']

    ems, f1s, qs, preds, answers = [], [], [], [], []
    for record in eval_dataset:
        answer = record['answer']
        record_aliases = aliases[record['answer_id']]
        record_answers = [answer] + record_aliases

        pred = model.query(record['question'], traversal=model_traversal, query_prompt=query_prompt)
        
        em, f1 = compute_em_f1(pred, record_answers)
        answers.append((record_answers))
        ems.append(em), f1s.append(f1), preds.append(pred) 
        qs.append(record['question'])
    return ems, f1s, qs, preds, answers


if __name__ == "__main__":
    max_tokens = 10000
    n_ctx = 1000
    model = utils.load_model(
        llm_model_path="./models/Meta-Llama-3-8B-Instruct.Q6_K.gguf",
        tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct"),
        n_ctx=n_ctx,
        max_tokens=max_tokens
    )

    evaluation([context], queries, expected_output)
    