import re
import json
import string
import time
from collections import defaultdict
from typing import List, Optional, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from curverag.curverag import CurveRAG

BASELINE_PROMPT = """Given the question: {questions} and the context: {context}, answer the questions.
    Reply with only the answer and nothing else. Do not include any additional text or explanation.
    Answer:
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

def evaluation_2wikimultihopqa(
        model: CurveRAG,
        dataset_path: str = '../datasets/2WikiMultihopQA/new/dev.json',
        dataset_size: int = 1000,
        alias_path: str = '../datasets/2WikiMultihopQA/new/id_aliases.json',
        model_traversal: str = 'all',
        query_prompt='generate_response_query'
    ):
    """
    Evaluates a CurveRAG model
    """
    print('------------- EVALUATION -------------')
    #print('query_prompt', query_prompt)
    with open(dataset_path, 'rb') as f:
        eval_dataset = json.load(f)
        eval_dataset = eval_dataset[:dataset_size]

    aliases = defaultdict(list)
    with open(alias_path, 'rb') as f:
        for line in f:
            record = json.loads(line)
            aliases[record['Q_id']] += record['aliases'] + record['demonyms']

    for record in eval_dataset:
        answer = record['answer']
        record_aliases = aliases[record['answer_id']]
        record_answers = [answer] + record_aliases
        result = {
            'question': record['question'],
            'answers': record_answers,
            'error': None,
            'em': None,
            'f1': None,
            'prediction': None,
            'graph': None,
            'attempts': 1
        }

        while True:
            try:
                pred, record_graph = model.query(record['question'], traversal=model_traversal, query_prompt=query_prompt)
                em, f1 = compute_em_f1(pred, record_answers)
                result.update({
                    'em': em,
                    'f1': f1,
                    'prediction': pred,
                    'graph': record_graph,
                    'attempts': result['attempts']
                })
                break  # Success, exit retry loop
            except Exception as e:
                result['error'] = str(e)
                if result['attempts'] >= 3:  # Max 3 attempts
                    result['error'] = f"Failed after 3 attempts. Last error: {str(e)}"
                    break
                
                # Wait before retry
                wait_time = 61  # seconds
                time.sleep(wait_time)
                result['attempts'] += 1
                result['error'] = f"Retry {result['attempts']}/3: {str(e)}"
        
        yield result
    

def evaluation_musique(
        model: CurveRAG,
        dataset_path: str = '../datasets/musique_data_v1.0/musique_ans_v1.0_dev.jsonl',
        dataset_size: int = 1000,
        model_traversal: str = 'all',
        query_prompt: str = 'generate_response_query'
    ):
    """
    Evaluates a CurveRAG model on the MuSiQue dataset.
    
    Args:
        model: The CurveRAG model instance
        dataset_path: Path to the MuSiQue JSONL file
        dataset_size: Maximum number of examples to evaluate
        model_traversal: Traversal strategy for the graph
        query_prompt: Name of the prompt template to use for queries
        
    Yields:
        Dictionary containing evaluation results for each example
    """
    print('------------- MUSIQUE EVALUATION -------------')
    print(f'Dataset path: {dataset_path}')
    
    # Load and process the MuSiQue dataset
    eval_dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if dataset_size and i >= dataset_size:
                break
                
            try:
                data = json.loads(line.strip())
                # Prepare answers (include both answer and its aliases)
                answer = data.get('answer', '')
                answer_aliases = data.get('answer_aliases', [])
                record_answers = [answer] + [a for a in answer_aliases if a != answer]
                
                eval_dataset.append({
                    'question': data.get('question', ''),
                    'answers': record_answers,
                    'id': data.get('id', str(i)),
                    'metadata': {
                        'title': data.get('title', ''),
                        'decomposition': data.get('decomposition', '')
                    }
                })
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i+1}: {e}")
                continue
    
    # Process each example
    for record in eval_dataset:
        result = {
            'question': record['question'],
            'answers': record['answers'],
            'error': None,
            'em': None,
            'f1': None,
            'prediction': None,
            'graph': None,
            'attempts': 1,
            'id': record['id'],
            'metadata': record.get('metadata', {})
        }
        
        while True:
            try:
                pred, record_graph = model.query(
                    record['question'], 
                    traversal=model_traversal, 
                    query_prompt=query_prompt
                )
                
                # Calculate metrics
                em, f1 = compute_em_f1(pred, record['answers'])
                
                result.update({
                    'em': em,
                    'f1': f1,
                    'prediction': pred,
                    'graph': record_graph,
                    'attempts': result['attempts']
                })
                break  # Success, exit retry loop
                
            except Exception as e:
                result['error'] = str(e)
                if result['attempts'] >= 3:  # Max 3 attempts
                    result['error'] = f"Failed after 3 attempts. Last error: {str(e)}"
                    break
                    
                # Wait before retry
                wait_time = 61  # seconds
                time.sleep(wait_time)
                result['attempts'] += 1
                result['error'] = f"Retry {result['attempts']}/3: {str(e)}"
        
        yield result


def evaluation_musique_baseline(
        openai_client,
        sentence_transformer_model_name: str = 'all-MiniLM-L6-v2',
        openai_model: str = 'gpt-4.1-mini',
        dataset_path: str = '../datasets/musique_data_v1.0/musique_ans_v1.0_dev.jsonl',
        dataset_size: int = 1000,
    ):
    """
    Evaluates a CurveRAG model on the MuSiQue dataset.
    
    Args:
        model: The CurveRAG model instance
        dataset_path: Path to the MuSiQue JSONL file
        dataset_size: Maximum number of examples to evaluate
        model_traversal: Traversal strategy for the graph
        query_prompt: Name of the prompt template to use for queries
        
    Yields:
        Dictionary containing evaluation results for each example
    """
    print('------------- MUSIQUE EVALUATION -------------')
    print(f'Dataset path: {dataset_path}')
    
    # Load and process the MuSiQue dataset
    sentence_model = SentenceTransformer(sentence_transformer_model_name)
        
    eval_dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if dataset_size and i >= dataset_size:
                break
                
            try:
                data = json.loads(line.strip())
                # Prepare answers (include both answer and its aliases)
                answer = data.get('answer', '')
                answer_aliases = data.get('answer_aliases', [])
                record_answers = [answer] + [a for a in answer_aliases if a != answer]
                
                eval_dataset.append({
                    'question': data.get('question', ''),
                    'answers': record_answers,
                    'context': ' '.join([d['paragraph_text'] for d in data['paragraphs']]),
                    'id': data.get('id', str(i)),
                    'metadata': {
                        'title': data.get('title', ''),
                        'decomposition': data.get('decomposition', '')
                    }
                })
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i+1}: {e}")
                continue

    question_embeddings = sentence_model.encode([record['question'] for record in eval_dataset])
    context_embeddings = sentence_model.encode([str(record['context']) for record in eval_dataset])
    
    # Process each example
    for i, record in enumerate(eval_dataset):
        sims = cosine_similarity([question_embeddings[i]], context_embeddings)[0]
        #print('sims', sims.shape, sims[0].shape, sims)
        # get top 5 context
        top_5_context = np.argsort(sims)[::-1][:5]
        print('top_5_context', top_5_context)
        context = ' '.join([str(eval_dataset[j]['context']) for j in top_5_context])
        # fill in the prompt
        prompt = BASELINE_PROMPT.format(
            questions=record['question'],
            context=context,
        )
        result = {
            'question': record['question'],
            'answers': record['answers'],
            'error': None,
            'em': None,
            'f1': None,
            'prediction': None,
            'graph': None,
            'attempts': 1,
            'id': record['id'],
            'metadata': record.get('metadata', {})
        }
        
        while True:
            try:
                pred = openai_client.responses.create(
                    model=openai_model,
                    input=prompt
                ).output_text
                
                # Calculate metrics
                em, f1 = compute_em_f1(pred, record['answers'])
                
                result.update({
                    'em': em,
                    'f1': f1,
                    'prediction': pred,
                    'attempts': result['attempts']
                })
                break  # Success, exit retry loop
                
            except Exception as e:
                result['error'] = str(e)
                if result['attempts'] >= 3:  # Max 3 attempts
                    result['error'] = f"Failed after 3 attempts. Last error: {str(e)}"
                    break
                    
                # Wait before retry
                wait_time = 61  # seconds
                time.sleep(wait_time)
                result['attempts'] += 1
                result['error'] = f"Retry {result['attempts']}/3: {str(e)}"
        
        yield result


def evaluation_2wikimultihopqa_baseline(
        openai_client,
        sentence_transformer_model_name: str = 'all-MiniLM-L6-v2',
        openai_model: str = 'gpt-4.1-mini',
        dataset_path: str = '../datasets/2WikiMultihopQA/new/dev.json',
        dataset_size: int = 1000,
        context_size: int = 1000,
        alias_path: str = '../datasets/2WikiMultihopQA/new/id_aliases.json',
    ):
    """
    Evaluates a CurveRAG model
    """
    print('------------- EVALUATION -------------')
    
    #print('query_prompt', query_prompt)
    with open(dataset_path, 'rb') as f:
        eval_dataset = json.load(f)
        eval_dataset_context = eval_dataset[:context_size]
        eval_dataset = eval_dataset[:dataset_size]
    print('eval_dataset', len(eval_dataset))
    print('eval_dataset_context', len(eval_dataset_context))
    aliases = defaultdict(list)
    with open(alias_path, 'rb') as f:
        for line in f:
            record = json.loads(line)
            aliases[record['Q_id']] += record['aliases'] + record['demonyms']

    sentence_model = SentenceTransformer(sentence_transformer_model_name)
    question_embeddings = sentence_model.encode([record['question'] for record in eval_dataset])
    context_embeddings = sentence_model.encode([str(record['context']).replace('[', '').replace(']', '. ').replace('\'', '') for record in eval_dataset_context])
    for i, record in enumerate(eval_dataset):
        answer = record['answer']
        record_aliases = aliases[record['answer_id']]
        record_answers = [answer] + record_aliases
        sims = cosine_similarity([question_embeddings[i]], context_embeddings)[0]
        #print('sims', sims.shape, sims[0].shape, sims)
        # get top 5 context
        top_5_context = np.argsort(sims)[::-1][:5]
        print('top_5_context', top_5_context)
        context = ' '.join([str(eval_dataset_context[j]['context']).replace('[', '').replace(']', '. ').replace('\'', '') for j in top_5_context])
        # fill in the prompt
        prompt = BASELINE_PROMPT.format(
            questions=record['question'],
            context=context,
        )
        result = {
            'question': record['question'],
            'answers': record_answers,
            'error': None,
            'em': None,
            'f1': None,
            'prediction': None,
            'attempts': 1
        }

        while True:
            try:
                pred = openai_client.responses.create(
                    model=openai_model,
                    input=prompt
                ).output_text

                em, f1 = compute_em_f1(pred, record_answers)
                result.update({
                    'em': em,
                    'f1': f1,
                    'prediction': pred,
                    'attempts': result['attempts']
                })
                break  # Success, exit retry loop
            except Exception as e:
                print('error', e)
                result['error'] = str(e)
                if result['attempts'] >= 3:  # Max 3 attempts
                    result['error'] = f"Failed after 3 attempts. Last error: {str(e)}"
                    break
                
                # Wait before retry
                wait_time = 61  # seconds
                time.sleep(wait_time)
                result['attempts'] += 1
                result['error'] = f"Retry {result['attempts']}/3: {str(e)}"
        
        yield result





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