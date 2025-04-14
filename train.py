"""Train Knowledge Graph embeddings for link prediction using AttH model."""

import json
import logging
import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim
import tqdm

import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from utils.train import get_savedir, avg_both, format_metrics, count_params

def train_step(model, batch, regularizer, n_entities, neg_sample_size=50, double_neg=True):
    """Run single training step."""
    # Generate negative samples
    if neg_sample_size > 0:
        # Positive samples forward pass
        positive_score, factors = model(batch)
        positive_score = F.logsigmoid(positive_score)
        
        # Generate and process negative samples
        neg_batch = batch.repeat(neg_sample_size, 1)
        neg_samples = torch.Tensor(np.random.randint(
            n_entities,
            size=batch.shape[0] * neg_sample_size)
        ).to(batch.dtype)
        neg_batch[:, 2] = neg_samples
        
        if double_neg:
            head_neg_samples = torch.Tensor(np.random.randint(
                n_entities,
                size=batch.shape[0] * neg_sample_size)
            ).to(batch.dtype)
            neg_batch[:, 0] = head_neg_samples
            
        negative_score, _ = model(neg_batch)
        negative_score = F.logsigmoid(-negative_score)
        loss = -torch.cat([positive_score, negative_score], dim=0).mean()
    else:
        # Without negative sampling
        predictions, factors = model(batch, eval_mode=True)
        truth = batch[:, 2]
        loss = F.cross_entropy(predictions, truth)

    # Add regularization
    loss += regularizer.forward(factors)
    return loss

def main():
    # Default hyperparameters (based on RotH WN18RR settings)
    class Args:
        def __init__(self):
            self.dataset = "WN18RR"
            self.model = "AttH"
            self.rank = 32
            self.regularizer = "N3"
            self.reg = 0.0
            self.optimizer = "Adam"
            self.max_epochs = 300
            self.patience = 15
            self.valid = 5
            self.batch_size = 500
            self.neg_sample_size = 50
            self.init_size = 0.001
            self.learning_rate = 0.001
            self.gamma = 0.0
            self.bias = "learn"
            self.dtype = "double"
            self.double_neg = True
            self.multi_c = True
            self.debug = False
            self.dropout = 0.0
    
    args = Args()
    save_dir = get_savedir(args.model, args.dataset)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log")
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))

    # Create dataset
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    # Load data
    logging.info("\t " + str(dataset.get_shape()))
    train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    # Save config
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)

    # Create model
    model = getattr(models, args.model)(args)
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    model.cuda()

    # Setup training
    regularizer = getattr(regularizers, args.regularizer)(args.reg)
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    counter = 0
    best_mrr = None
    best_epoch = None
    n_entities = args.sizes[0]
    
    logging.info("\t Start training")
    for step in range(args.max_epochs):
        # Train step
        model.train()
        total_loss = 0
        batch_counter = 0
        
        # Shuffle training data
        actual_examples = train_examples[torch.randperm(train_examples.shape[0]), :]
        
        # Process batches
        with tqdm.tqdm(total=train_examples.shape[0], unit='ex', disable=False) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            
            while b_begin < train_examples.shape[0]:
                batch = actual_examples[b_begin:b_begin + args.batch_size].cuda()
                
                # Compute loss and update
                loss = train_step(
                    model, batch, regularizer, n_entities,
                    args.neg_sample_size, args.double_neg
                )
                
                optim_method.zero_grad()
                loss.backward()
                optim_method.step()
                
                b_begin += args.batch_size
                total_loss += loss.item()
                batch_counter += 1
                bar.update(batch.shape[0])
                bar.set_postfix(loss=f'{loss.item():.4f}')
                
        avg_loss = total_loss / batch_counter
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, avg_loss))

        # Validation step
        model.eval()
        valid_loss = 0
        valid_counter = 0
        b_begin = 0
        
        while b_begin < valid_examples.shape[0]:
            batch = valid_examples[b_begin:b_begin + args.batch_size].cuda()
            with torch.no_grad():
                loss = train_step(
                    model, batch, regularizer, n_entities,
                    args.neg_sample_size, args.double_neg
                )
            valid_loss += loss.item()
            valid_counter += 1
            b_begin += args.batch_size
            
        avg_valid_loss = valid_loss / valid_counter
        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, avg_valid_loss))

        if (step + 1) % args.valid == 0:
            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
            logging.info(format_metrics(valid_metrics, split="valid"))

            valid_mrr = valid_metrics["MRR"]
            if not best_mrr or valid_mrr > best_mrr:
                best_mrr = valid_mrr
                counter = 0
                best_epoch = step
                logging.info("\t Saving model at epoch {} in {}".format(step, save_dir))
                torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                model.cuda()
            else:
                counter += 1
                if counter == args.patience:
                    logging.info("\t Early stopping")
                    break

    logging.info("\t Optimization finished")
    if not best_mrr:
        torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
    else:
        logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
        model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    model.cuda()
    model.eval()

    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
    logging.info(format_metrics(valid_metrics, split="valid"))

    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    logging.info(format_metrics(test_metrics, split="test"))

if __name__ == "__main__":
    main()