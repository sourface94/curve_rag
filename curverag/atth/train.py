"""Train Knowledge Graph embeddings for link prediction."""

import argparse
import json
import logging
import os

import toml
import torch
import torch.optim

from curverag.atth import models
import curverag.atth.optimizers.regularizers as regularizers
from curverag.atth.kg_dataset import KGDataset
from curverag.atth.optimizers.kg_optimizer import KGOptimizer
from curverag.atth.models.hyperbolic import AttH
from curverag.atth.utils.train import get_savedir, avg_both, format_metrics, count_params


def train(dataset):
    with open('config.toml', 'r') as f:
        config = toml.load(f)

    save_dir = get_savedir('AttH', dataset.name)

    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log")
    )

    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))

    # get dataset splits
    sizes = dataset.get_shape()
    logging.info("\t " + str(dataset.get_shape()))
    train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    config['model']['sizes'] = dataset.get_shape()

    # save config
    with open(os.path.join(save_dir, "config.toml"), "w") as f:
        toml.dump(config, f)

    # create model
    model = AttH(**config['model'])
    
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    device = "cuda"
    model#.to(device)

    # get optimizer
    regularizer = getattr(regularizers, config['train']['regularizer'])(config['train']['reg'])
    optim_method = getattr(torch.optim, config['train']['optimizer'])(model.parameters(), lr=config['train']['learning_rate'])
    optimizer = KGOptimizer(model, regularizer, optim_method, config['train']['batch_size'], config['train']['neg_sample_size'],
                            bool(config['train']['double_neg']))
    counter = 0
    best_mrr = None
    best_epoch = None
    logging.info("\t Start training")
    max_epochs = config['train']['max_epochs']
    valid = config['train']['valid']
    patience = config['train']['patience']
    for step in range(max_epochs):
        # Train step
        model.train()
        train_loss = optimizer.epoch(train_examples)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

        # Valid step
        model.eval()
        valid_loss = optimizer.calculate_valid_loss(valid_examples)
        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

        if (step + 1) % valid == 0:
            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
            logging.info(format_metrics(valid_metrics, split="valid"))

            valid_mrr = valid_metrics["MRR"]
            if not best_mrr or valid_mrr > best_mrr:
                best_mrr = valid_mrr
                counter = 0
                best_epoch = step
                logging.info("\t Saving model at epoch {} in {}".format(step, save_dir))
                torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                model#.cuda()
            else:
                counter += 1
                if counter == patience:
                    logging.info("\t Early stopping")
                    break
                elif counter == patience // 2:
                    pass
                    # logging.info("\t Reducing learning rate")
                    # optimizer.reduce_lr()

    logging.info("\t Optimization finished")
    if not best_mrr:
        torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
    else:
        logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
        model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    model#.cuda()
    model.eval()

    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
    logging.info(format_metrics(valid_metrics, split="valid"))

    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    logging.info(format_metrics(test_metrics, split="test"))

    return model


if __name__ == "__main__":
    dataset_path = './data/medical_docs'
    dataset = KGDataset(dataset_path, debug=False, name='medical_docs')
    train(dataset)
