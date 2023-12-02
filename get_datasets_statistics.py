import numpy as np
import os
import argparse
import json
from utils import *


def main(args):
    split = args.split_type
    dataset_name = args.dataset_name
    path = args.path
    save_format = args.save_format
    save_path = args.save_path

    # Get path to dataset
    dataset_path = os.path.join(path, dataset_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load kg
    if split == "train":
        kg = np.loadtxt(os.path.join(dataset_path, "train2id.txt"), dtype=int)

    entity2id = read_entity2id(dataset_path)
    relation2id = read_relation2id(dataset_path)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Load entity names if exist
    if os.path.exists(os.path.join(dataset_path, "id2name.txt")):
        id2name = {}
        name2id = {}
        with open(
            os.path.join(dataset_path, "id2name.txt"), "r", encoding="utf-8"
        ) as file:
            for line in file:
                id, name = line.strip().split("\t")
                id2name[int(id)] = name
                name2id[name] = int(id)

    # Load classes & entity -> classes
    if dataset_name in ["KG20C"]:
        ent2classes2id = load_json(dataset_path, f"entity2class2id")
        ent2classes = load_json(dataset_path, f"entity2class")
    else:
        ent2classes2id = load_json(dataset_path, f"entity2classes2id")
        ent2classes = load_json(dataset_path, f"entity2classes")

    class2id = load_json(dataset_path, f"class2id")
    id2class = {v: k for k, v in class2id.items()}

    ### STATISTICS ###
    # Frequency of relations
    rel_freq = {}
    for rel in kg[:, 1]:
        if id2relation[rel] in rel_freq:
            rel_freq[id2relation[rel]] += 1
        else:
            rel_freq[id2relation[rel]] = 1

    # order by frequency
    rel_freq = {
        k: v
        for k, v in sorted(rel_freq.items(), key=lambda item: item[1], reverse=True)
    }

    # Frequency of entities
    ent_freq = {}
    for ent in kg[:, 0]:
        if id2entity[ent] in ent_freq:
            ent_freq[id2entity[ent]] += 1
        else:
            ent_freq[id2entity[ent]] = 1

    # order by frequency
    ent_freq = {
        k: v
        for k, v in sorted(ent_freq.items(), key=lambda item: item[1], reverse=True)
    }

    # Frequency of classes
    class_freq = {}
    for ent, classes in ent2classes.items():
        for class_ in classes:
            if class_ in class_freq:
                class_freq[class_] += 1
            else:
                class_freq[class_] = 1

    # order by frequency
    class_freq = {
        k: v
        for k, v in sorted(class_freq.items(), key=lambda item: item[1], reverse=True)
    }

    # Average number of triples in 1-hop neighborhood
    # taking into account head/tail appearance
    # also storing it in dicts for each entity
    avg_triples_1hop = 0
    entity2triples_1hop = {}
    entity2unique_entities_1hop = {}
    entity2nb_unique_entities_1hop = {}
    entity2unique_relations_1hop = {}
    entity2nb_unique_relations_1hop = {}
    entity2unique_classes_1hop = {}
    entity2nb_unique_classes_1hop = {}
    for entID in entity2id.values():
        try:
            entity = id2name[entID]
            entity2unique_entities_1hop[entity] = set()
            entity2unique_relations_1hop[entity] = set()
            entity2unique_classes_1hop[entity] = set()
        except:
            entity = id2entity[entID]
            entity2unique_entities_1hop[entity] = set()
            entity2unique_relations_1hop[entity] = set()
            entity2unique_classes_1hop[entity] = set()
        # get 1-hop neighborhood where ent is head or tail
        triples_1hop = kg[np.where((kg[:, 0] == entID) | (kg[:, 2] == entID))]
        # count number of triples
        entity2triples_1hop[entity] = len(triples_1hop)
        # unique entities
        entity2unique_entities_1hop[entity] = set(triples_1hop[:, 0]).union(
            set(triples_1hop[:, 2])
        )
        # count number of unique entities
        entity2nb_unique_entities_1hop[entity] = len(
            entity2unique_entities_1hop[entity]
        )
        # convert using either id2name or id2entity
        try:
            entity2unique_entities_1hop[entity] = list(
                set([id2name[ent] for ent in entity2unique_entities_1hop[entity]])
            )
        except:
            entity2unique_entities_1hop[entity] = list(
                set([id2entity[ent] for ent in entity2unique_entities_1hop[entity]])
            )
        # unique relations
        entity2unique_relations_1hop[entity] = set(triples_1hop[:, 1])
        # count number of unique relations
        entity2nb_unique_relations_1hop[entity] = len(
            entity2unique_relations_1hop[entity]
        )
        # convert using id2relation
        entity2unique_relations_1hop[entity] = list(
            set([id2relation[rel] for rel in entity2unique_relations_1hop[entity]])
        )
        # unique classes in the neighborhood of ent
        for ent_ in entity2unique_entities_1hop[entity]:
            # if the entity has classes
            if ent_ in ent2classes:
                entity2unique_classes_1hop[entity].update(ent2classes[ent_])
        entity2unique_classes_1hop[entity] = list(entity2unique_classes_1hop[entity])
        # count number of unique classes
        entity2nb_unique_classes_1hop[entity] = len(entity2unique_classes_1hop[entity])

    # Get summary statistics based on the previous dicts
    avg_triples_1hop = np.mean(list(entity2triples_1hop.values()))
    avg_unique_entities_1hop = np.mean(list(entity2nb_unique_entities_1hop.values()))
    avg_unique_relations_1hop = np.mean(list(entity2nb_unique_relations_1hop.values()))
    avg_unique_classes_1hop = np.mean(list(entity2nb_unique_classes_1hop.values()))

    ### STORE EVERYTHING IN A JSON FILE ###
    # Create dict
    statistics = {}
    statistics["dataset_name"] = dataset_name
    statistics["split"] = split
    statistics["num_entities"] = len(entity2id)
    statistics["num_relations"] = len(relation2id)
    statistics["num_classes"] = len(class2id)
    statistics["num_triples"] = len(kg)
    statistics["avg_triples_1hop"] = avg_triples_1hop
    statistics["avg_unique_entities_1hop"] = avg_unique_entities_1hop
    statistics["avg_unique_relations_1hop"] = avg_unique_relations_1hop
    statistics["avg_unique_classes_1hop"] = avg_unique_classes_1hop
    statistics["rel_freq"] = rel_freq
    statistics["ent_freq"] = ent_freq
    statistics["class_freq"] = class_freq
    statistics["entity2triples_1hop"] = entity2triples_1hop
    statistics["entity2unique_entities_1hop"] = entity2unique_entities_1hop
    statistics["entity2nb_unique_entities_1hop"] = entity2nb_unique_entities_1hop
    statistics["entity2unique_relations_1hop"] = entity2unique_relations_1hop
    statistics["entity2nb_unique_relations_1hop"] = entity2nb_unique_relations_1hop
    statistics["entity2unique_classes_1hop"] = entity2unique_classes_1hop
    statistics["entity2nb_unique_classes_1hop"] = entity2nb_unique_classes_1hop

    # Save dict
    if save_format == "json":
        with open(
            os.path.join(save_path, f"{dataset_name}_{split}_statistics.json"), "w"
        ) as file:
            json.dump(statistics, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Arguments with defaults
    parser.add_argument(
        "-d", "--dataset_name", type=str, default="codex-s", help="Name of the dataset"
    )
    parser.add_argument(
        "-s", "--split_type", type=str, default="train", help="Type of data split"
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="preprocessed_datasets",
        help="Path to preprocessed datasets",
    )
    parser.add_argument(
        "-f",
        "--save_format",
        type=str,
        default="json",
        help="Format of file to save statistics",
    )
    parser.add_argument(
        "-sp",
        "--save_path",
        type=str,
        default="datasets_statistics",
        help="Path to save statistics",
    )
    parser.add_argument("-v", "--verbose", type=bool, default=True, help="Verbose mode")

    # Parse the arguments
    args = parser.parse_args()

    # Call main function
    main(args)
