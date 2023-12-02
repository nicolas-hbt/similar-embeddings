import numpy as np
from utils import *
from collections import Counter
import os
import argparse


def main(args):
    split = args.split_type
    dataset_name = args.dataset_name
    path = args.path
    dataset_path = os.path.join(path, dataset_name)

    kg = np.loadtxt(os.path.join(dataset_path, f"train2id.txt"), dtype=int)

    class2id = load_json(dataset_path, f"class2id")
    id2class = {v: k for k, v in class2id.items()}

    relation2id = read_relation2id(dataset_path)
    id2relation = {int(id): relation for relation, id in relation2id.items()}

    # Load classes & entity -> classes
    if dataset_name in ["KG20C"]:
        ent2classes2id = load_json(dataset_path, f"entity2class2id")
        ent2classes = load_json(dataset_path, f"entity2class")
        class2entities = load_json(dataset_path, f"class2entity")
        class2entities2id = load_json(dataset_path, f"class2entity2id")
    else:
        ent2classes2id = load_json(dataset_path, f"entity2classes2id")
        ent2classes = load_json(dataset_path, f"entity2classes")
        class2entities = load_json(dataset_path, f"class2entities")
        class2entities2id = load_json(dataset_path, f"class2entities2id")

    one_hop = get_all_1_hops(kg)

    unique_relation_ids = set(relation2id.values())
    unique_class_ids = set(class2entities2id.keys())

    rel_per_class = {int(class_id): {} for class_id in unique_class_ids}
    rel_per_class_labels = {
        id2class[int(class_id)]: {}
        for class_id in unique_class_ids
        if int(class_id) in id2class
    }

    # Iterate through entIDs
    for entID in one_hop:
        classesID = ent2classes2id.get(str(entID), [])
        rels = one_hop[entID][:, [1]].flatten()
        rel_count = Counter(rels)

        for classID in classesID:
            if classID in rel_per_class:
                # Update rel_per_class with the accumulated counts
                for relation_id, count in rel_count.items():
                    if int(relation_id) not in rel_per_class[classID]:
                        rel_per_class[int(classID)][int(relation_id)] = 0
                    rel_per_class[int(classID)][int(relation_id)] += count

    # Iterate through rel_per_class to create rel_per_class_labels
    for class_id, rel_counts in rel_per_class.items():
        if int(class_id) not in id2class:
            continue
        class_label = id2class[int(class_id)]

        for relation_id, count in rel_counts.items():
            relation_label = id2relation[relation_id]
            if count != 0:
                rel_per_class_labels[class_label][relation_label] = count

    # order inner dicts in rel_per_class per count
    for class_id, rel_counts in rel_per_class.items():
        rel_per_class[class_id] = {
            k: v
            for k, v in sorted(
                rel_counts.items(), key=lambda item: item[1], reverse=True
            )
        }

    # order inner dicts in rel_per_class_labels per count
    for class_label, rel_counts in rel_per_class_labels.items():
        rel_per_class_labels[class_label] = {
            k: v
            for k, v in sorted(
                rel_counts.items(), key=lambda item: item[1], reverse=True
            )
        }

    save_json(rel_per_class, os.path.join(dataset_path, "rel_per_class2id.json"))
    save_json(
        rel_per_class_labels, os.path.join(dataset_path, "rel_per_class_labels.json")
    )


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

    # Parse the arguments
    args = parser.parse_args()

    # Call main function
    main(args)
