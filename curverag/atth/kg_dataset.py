"""Dataset class for loading and processing KG datasets."""

import os
import pickle as pkl

import numpy as np
import torch


class KGDataset(object):
    """Knowledge Graph dataset class."""

    def __init__(self, data_path, debug, name='name'):
        """Creates KG dataset object for data loading.

        Args:
             data_path: Path to directory containing train/valid/test pickle files produced by process.py
             debug: boolean indicating whether to use debug mode or not
             if true, the dataset will only contain 1000 examples for debugging.
        """
        self.data_path = data_path
        self.debug = debug
        self.data = {}
        for split in ["train", "test", "valid"]:
            file_path = os.path.join(self.data_path, split + ".pickle")
            with open(file_path, "rb") as in_file:
                self.data[split] = pkl.load(in_file)
        filters_file = open(os.path.join(self.data_path, "to_skip.pickle"), "rb")
        self.to_skip = pkl.load(filters_file)
        filters_file.close()
        # Get unique entity and predicate indices from head, relation, and tail columns
        train_data = self.data["train"]
        valid_data = self.data["valid"]
        test_data = self.data["test"]
        unique_entities = set(train_data[:, 0]).union(set(train_data[:, 2])).union(set(valid_data[:, 0])).union(set(valid_data[:, 2])).union(set(test_data[:, 0])).union(set(test_data[:, 2]))
        unique_predicates = set(train_data[:, 1]).union(set(valid_data[:, 1])).union(set(test_data[:, 1]))
        
        self.n_entities = len(unique_entities)
        self.n_predicates = len(unique_predicates) * 2  # Multiply by 2 for inverse relations
        self.name = name

        try:
            file_path = os.path.join(self.data_path, "nodes_id_idx.pickle")
            with open(file_path, "rb") as in_file:
                self.nodes_id_idx = pkl.load(in_file)
            file_path = os.path.join(self.data_path, "relationship_name_idx.pickle")
            with open(file_path, "rb") as in_file:
                self.relationship_name_idx = pkl.load(in_file)
        except:
            pass


    def get_examples(self, split, rel_idx=-1):
        """Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relation)

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """
        examples = self.data[split]
        if split == "train":
            copy = np.copy(examples)
            tmp = np.copy(copy[:, 0])
            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] += self.n_predicates // 2
            examples = np.vstack((examples, copy))
        if rel_idx >= 0:
            examples = examples[examples[:, 1] == rel_idx]
        if self.debug:
            examples = examples[:1000]
        return torch.from_numpy(examples.astype("int64"))

    def get_filters(self, ):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.to_skip

    def get_shape(self):
        """Returns KG dataset shape."""
        return self.n_entities, self.n_predicates, self.n_entities
