from utils import *

import pickle
import json
import torch
from typing import List
import numpy as np
import pykeen.nn
import os
import argparse
from tqdm import tqdm
import time


def retrieve_dict(args, filename, MODELS, class2entities):
    # read json file if exists
    if os.path.exists(filename):
        with open(filename, "r") as json_file:
            return json.load(json_file)
    else:
        print("Dict initialization...")
        return init_dict(MODELS, class2entities)


def init_dict(MODELS, class2entities):
    # initialize dict
    dict = {}
    for model in MODELS:
        dict[model] = {}
        for cl in class2entities.keys():
            dict[model][str(cl)] = {}
            dict[model][str(cl)]["num_entities"] = None
            dict[model][str(cl)]["rbo3"] = None
            dict[model][str(cl)]["rbo5"] = None
            dict[model][str(cl)]["rbo10"] = None
            dict[model][str(cl)]["rbo50"] = None
            dict[model][str(cl)]["rbo100"] = None
    return dict


def check_dict_init(dict):
    # Extract the keys from the first model's first class to use as a reference
    first_model = next(iter(dict))
    reference_classes = set(dict[first_model].keys())

    # Iterate over each model
    for model in dict:
        # Extract the current model's classes
        current_classes = set(dict[model].keys())
        if current_classes != reference_classes:
            return False

        # Iterate over each class in the model
        for class_key in dict[model]:
            # Extract the current class's rbo values
            current_rbo_values = set(dict[model][class_key].keys())
            if current_rbo_values != {"rbo3", "rbo5", "rbo10", "rbo50", "rbo100"}:
                return False

    return True


def check_dict_completion(dict):
    # Iterate over each model, class, and rbo value using list comprehension
    incomplete_values = [
        (model, class_key)
        for model in dict
        for class_key in dict[model]
        if any(value is None for value in dict[model][class_key].values())
    ]

    # Return the first incomplete value found, or None if all values are complete
    return incomplete_values[0] if incomplete_values else None


# Get the current timestamp as a string
timestamp = str(int(time.time()))

DATASET_DIR = "preprocessed_datasets"
MODELS = [
    "rdf2vec",
    "boxe",
    "conve",
    "distmult",
    "rescal",
    "transd",
    "transe",
    "tucker",
]


def main(args, MODELS):
    dataset_name = args.dataset_name
    dataset_path = os.path.join(DATASET_DIR, dataset_name)
    K = args.topk
    hops = args.hops
    save_path = args.save_path
    verbose = args.verbose
    split_type = args.split
    interval = args.interval

    # Define the filename with the timestamp
    filename = f"logs/{dataset_name}/expe4/Kmax={K}_hops={hops}.json"
    if not os.path.exists(f"logs/{dataset_name}/expe4"):
        os.makedirs(f"logs/{dataset_name}/expe4")

    # Load the dataset
    kg = np.loadtxt(os.path.join(dataset_path, f"{split_type}2id.txt"), dtype=int)
    entity2id = read_entity2id(dataset_path)
    # Map id -> name
    if (
        dataset_name == "codex-s"
        or dataset_name == "codex-m"
        or dataset_name == "FB15K-237"
    ):
        id2name = {}
        name2id = {}

        file_path = os.path.join(dataset_path, "id2name.txt")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    id, name = line.strip().split("\t")
                    id2name[int(id)] = name
                    name2id[name] = int(id)
        else:
            print(f"Warning: {file_path} does not exist.")
    entity2id = read_entity2id(dataset_path)
    id2entity = {int(id): entity for entity, id in entity2id.items()}

    relation2id = read_relation2id(dataset_path)
    id2relation = {int(id): relation for relation, id in relation2id.items()}
    # Load classes & entity -> classes
    if dataset_name in ["KG20C"]:
        ent2classes2id = load_json(dataset_path, f"entity2class2id")
        ent2classes = load_json(dataset_path, f"entity2class")
        class2ent = load_json(dataset_path, f"class2entity")
        class2ent2id = load_json(dataset_path, f"class2entity2id")
    else:
        ent2classes2id = load_json(dataset_path, f"entity2classes2id")
        ent2classes = load_json(dataset_path, f"entity2classes")
        class2ent = load_json(dataset_path, f"class2entities")
        class2ent2id = load_json(dataset_path, f"class2entities2id")

    class2id = load_json(dataset_path, f"class2id")
    id2class = {v: k for k, v in class2id.items()}

    rbo_dict = retrieve_dict(args, filename, MODELS, class2ent)
    if check_dict_init(rbo_dict):
        # check from where to resume
        mdl, class_key = check_dict_completion(rbo_dict)
        print(f"Resuming from {mdl}, {class_key}")
        # crop MODELS to start from mdl
        MODELS = MODELS[MODELS.index(mdl) :]

        # get first class in class2ent and if it does not match class_key, then the entries for the first model are not empty
        first_class = next(iter(class2ent))
        if first_class != class_key:
            token_resume = 1
        else:
            token_resume = 0

        # Load the entity embeddings
        for model_name in MODELS:
            model_path = os.path.join(
                "embeddings", dataset_name, f"{model_name}", "trained_model.pkl"
            )
            if model_name != "rdf2vec":
                try:
                    model = torch.load(model_path, map_location=torch.device("cpu"))
                    entity_representation_modules: List[
                        "pykeen.nn.Representation"
                    ] = model.entity_representations
                    entity_embeddings: pykeen.nn.Embedding = (
                        entity_representation_modules[0]
                    )
                    if model_name == "transd" and dataset_name == "FB15K-237":
                        entity_embedding_tensor = (
                            entity_embeddings._embeddings.weight.cpu().detach()
                        )
                    else:
                        entity_embedding_tensor: torch.FloatTensor = (
                            entity_embeddings().cpu().detach()
                        )
                    # Map pykeenID -> myIDs (useless for RDF2vec)
                    pykeen2id = parse_pykeen_dict(
                        f"embeddings/{dataset_name}/transe/training_triples/entity_to_id.tsv.gz",
                        format="dict",
                    )
                    id2pykeen = {v: k for k, v in pykeen2id.items()}
                    if dataset_name == "FB15K-237":
                        id2pykeen = dict(sorted(id2pykeen.items()))
                except:
                    try:
                        model = torch.load(model_path, map_location=torch.device("cpu"))
                        entity_embedding_tensor = (
                            model.entity_embeddings._embeddings.weight.cpu().detach()
                        )
                    except:
                        print(
                            f"Model {model_name} not found for dataset {dataset_name}"
                        )
                        continue
            else:
                with open(model_path, "rb") as file:
                    entity_embeddings = pickle.load(file)
                entity_embedding_tensor = (
                    torch.tensor(np.array(entity_embeddings)).cpu().detach()
                )
                entities = pd.read_csv(
                    os.path.join("embeddings", dataset_name, "rdf2vec", "entities.txt"),
                    sep="\t",
                    header=None,
                )
                relations = pd.read_csv(
                    os.path.join(
                        "embeddings", dataset_name, "rdf2vec", "relations.txt"
                    ),
                    sep="\t",
                    header=None,
                )
                entities2id_rdf2vec = {
                    entity: id for id, entity in enumerate(entities[0])
                }

                if dataset_name == "FB15K-237":
                    # create a dict mapping ids from entity2id_rdf2vec to the ids in entity2id
                    idrdf2vec2id = {}
                    for entity, idx in entities2id_rdf2vec.items():
                        idx_original = entity2id[entity]
                        idrdf2vec2id[idx] = idx_original
                    id_original2id_rdf2vec = {v: k for k, v in idrdf2vec2id.items()}
                    # create entityrdf2vec2name
                    rdf2vec2name = {}
                    rdf2vec2id2name = {}
                    for entity, idx in entities2id_rdf2vec.items():
                        try:
                            rdf2vec2name[entity] = id2name[idrdf2vec2id[idx]]
                            rdf2vec2id2name[idx] = id2name[idrdf2vec2id[idx]]
                        except:
                            continue

            try:
                pykeen2id = parse_pykeen_dict(
                    f"embeddings/{dataset_name}/transe/training_triples/entity_to_id.tsv.gz",
                    format="dict",
                )
                id2pykeen = {v: k for k, v in pykeen2id.items()}
                if dataset_name == "FB15K-237":
                    id2pykeen = dict(sorted(id2pykeen.items()))
            except:
                print("Pykeen IDs not found.")

            if dataset_name == "FB15K-237" or model_name == "rdf2vec":
                similarity_matrix = all_pairs_cosine_similarity(entity_embedding_tensor)
            else:
                original_np_array = entity_embedding_tensor.numpy()
                rearranged_np_array = np.zeros_like(original_np_array)
                original_np_array = entity_embedding_tensor.numpy()
                for old_index, new_index in pykeen2id.items():
                    rearranged_np_array[new_index] = original_np_array[old_index]
                # Convert the rearranged NumPy array back to a PyTorch tensor
                rearranged_tensor = torch.Tensor(rearranged_np_array)
                similarity_matrix = all_pairs_cosine_similarity(rearranged_tensor)

            # Set diagonal to a small value to exclude it from top K
            similarity_matrix.fill_diagonal_(float("-inf"))

            # Get the top K indices for each row
            topk_values, topk_indices = torch.topk(similarity_matrix, K, dim=1)

            if dataset_name == "FB15K-237" and model_name != "rdf2vec":
                contiguous2myIDS = {}
                myIDs2contiguous = {}
                contiguous2pykeenIDS = {}
                pykeenIDS2contiguous = {}
                new_topk_indices = np.zeros_like(topk_indices)
                i = 0

                for myID, pykeenID in id2pykeen.items():
                    contiguous2myIDS[i] = myID
                    myIDs2contiguous[myID] = i
                    contiguous2pykeenIDS[i] = pykeenID
                    pykeenIDS2contiguous[pykeenID] = i
                    new_topk_indices[i] = topk_indices[pykeenID]
                    new_topk_indices[i] = [
                        pykeen2id[old_index] for old_index in new_topk_indices[i]
                    ]
                    i += 1

                topk_indices = torch.tensor(new_topk_indices)

            graph_sim = load_json(
                dataset_path, f"fast_jaccard2id_{split_type}_K={K}_hops={hops}"
            )

            filtered_graph_sim = {
                entID: inner_dict
                for entID, inner_dict in graph_sim.items()
                if all(
                    jaccard_val != 0.0 for jaccard_val in list(inner_dict.values())[:3]
                )
            }
            # order the keys
            filtered_graph_sim = dict(
                sorted(filtered_graph_sim.items(), key=lambda x: int(x[0]))
            )

            if model_name == "rdf2vec" and dataset_name == "FB15K-237":
                keys_list = [
                    [int(key) for key in inner_dict.keys()]
                    for inner_dict in filtered_graph_sim.values()
                ]
                topk_indices_graph_based = torch.tensor(keys_list)
                to_keep_original = [int(entID) for entID in filtered_graph_sim.keys()]
                # map the keys and values using id_original2id_rdf2vec
                filtered_graph_sim_copy = filtered_graph_sim.copy()
                filtered_graph_sim = {}
                for key, value in filtered_graph_sim_copy.items():
                    filtered_graph_sim[str(id_original2id_rdf2vec[int(key)])] = {
                        id_original2id_rdf2vec[int(k)]: v for k, v in value.items()
                    }

            keys_list = [
                [int(key) for key in inner_dict.keys()]
                for inner_dict in filtered_graph_sim.values()
            ]

            # Create a PyTorch tensor from the list of keys
            topk_indices_graph_based = torch.tensor(keys_list)

            to_keep = [int(entID) for entID in filtered_graph_sim.keys()]

            if dataset_name == "FB15K-237" and model_name != "rdf2vec":
                to_keep_original = to_keep.copy()
                to_keep = [myIDs2contiguous[entID] for entID in to_keep]

            topk_values, topk_indices_embedding_based = (
                topk_values[to_keep],
                topk_indices[to_keep],
            )

            for class_name in class2ent.keys():
                start_time = time.time()

                if class_name == class_key and model_name == mdl:
                    token_resume = 0
                    print(f"Resuming from {class_name} with model {model_name}")
                if token_resume == 1:
                    continue

                classID = class2id[class_name]
                (
                    compute_rbo3,
                    compute_rbo5,
                    compute_rbo10,
                    compute_rbo50,
                    compute_rbo100,
                ) = (True, True, True, True, True)
                # Get the entities for this class
                entitiesID = class2ent2id[str(classID)]
                if dataset_name == "FB15K-237":
                    entitiesID = [
                        int(entID)
                        for entID in entitiesID
                        if int(entID) in to_keep_original
                    ]
                    if model_name == "rdf2vec":
                        entitiesID = [
                            id_original2id_rdf2vec[entID] for entID in entitiesID
                        ]
                else:
                    entitiesID = [
                        int(entID) for entID in entitiesID if int(entID) in to_keep
                    ]

                number_of_entities = len(entitiesID)
                rbo_dict[model_name][str(class_name)][
                    "num_correct_entities"
                ] = number_of_entities

                # pass to the next class if there are not enough entities for this class
                if (
                    number_of_entities < 3
                    and model_name == MODELS[MODELS.index(mdl) + 1]
                ):
                    print(
                        f"Skipping class {class_name} since it has {number_of_entities} correct entities"
                    )
                    rbo_dict[model_name][str(class_name)].pop("rbo100")
                    rbo_dict[model_name][str(class_name)].pop("rbo50")
                    rbo_dict[model_name][str(class_name)].pop("rbo10")
                    rbo_dict[model_name][str(class_name)].pop("rbo5")
                    rbo_dict[model_name][str(class_name)].pop("rbo3")
                    continue

                # determine what RBO@K to compute
                if number_of_entities < 100:
                    compute_rbo100 = False
                    # delete the corresponding keys in the dict
                    rbo_dict[model_name][str(class_name)].pop("rbo100")
                    if number_of_entities < 50:
                        compute_rbo50 = False
                        # delete the corresponding keys in the dict
                        rbo_dict[model_name][str(class_name)].pop("rbo50")
                        if number_of_entities < 10:
                            compute_rbo10 = False
                            # delete the corresponding keys in the dict
                            rbo_dict[model_name][str(class_name)].pop("rbo10")
                            if number_of_entities < 5:
                                compute_rbo5 = False
                                # delete the corresponding keys in the dict
                                rbo_dict[model_name][str(class_name)].pop("rbo5")
                                if number_of_entities < 3:
                                    compute_rbo3 = False
                                    # delete the corresponding keys in the dict
                                    rbo_dict[model_name][str(class_name)].pop("rbo3")

                # initialize the RBO values at 0.0 for the remaining entries
                if compute_rbo3:
                    rbo_dict[model_name][str(class_name)]["rbo3"] = 0.0
                if compute_rbo5:
                    rbo_dict[model_name][str(class_name)]["rbo5"] = 0.0
                if compute_rbo10:
                    rbo_dict[model_name][str(class_name)]["rbo10"] = 0.0
                if compute_rbo50:
                    rbo_dict[model_name][str(class_name)]["rbo50"] = 0.0
                if compute_rbo100:
                    rbo_dict[model_name][str(class_name)]["rbo100"] = 0.0

                cpt_rbo100, cpt_rbo50, cpt_rbo10, cpt_rbo5, cpt_rbo3 = 0, 0, 0, 0, 0

                for entID in entitiesID:
                    number_non_zero = len(
                        [
                            jaccard_val
                            for jaccard_val in filtered_graph_sim[str(entID)].values()
                            if jaccard_val != 0.0
                        ]
                    )

                    if compute_rbo100 and number_non_zero >= 100:
                        cpt_rbo100 += 1
                        # get the top 100 neareast neighbors wrt Jaccard
                        top100 = list(filtered_graph_sim[str(entID)].keys())
                        top100 = np.array([int(entID) for entID in top100])
                        # get the top 100 nearest neighbors wrt embedding
                        top100_emb = (
                            topk_indices[entID].numpy()[:100]
                            if dataset_name != "FB15K-237" or model_name == "rdf2vec"
                            else topk_indices[myIDs2contiguous[entID]].numpy()[:100]
                        )
                        # Perform the row-wise comparison
                        rbo_class100 = rbo_score(top100, top100_emb)
                        rbo_dict[model_name][str(class_name)]["rbo100"] += rbo_class100

                    if compute_rbo50 and number_non_zero >= 50:
                        cpt_rbo50 += 1
                        # get the top 50 neareast neighbors wrt Jaccard
                        top50 = list(filtered_graph_sim[str(entID)].keys())[:50]
                        top50 = np.array([int(entID) for entID in top50])
                        # get the top 50 nearest neighbors wrt embedding
                        top50_emb = (
                            topk_indices[entID].numpy()[:50]
                            if dataset_name != "FB15K-237" or model_name == "rdf2vec"
                            else topk_indices[myIDs2contiguous[entID]].numpy()[:50]
                        )
                        # Perform the row-wise comparison
                        rbo_class50 = rbo_score(top50, top50_emb)
                        rbo_dict[model_name][str(class_name)]["rbo50"] += rbo_class50

                    if compute_rbo10 and number_non_zero >= 10:
                        cpt_rbo10 += 1
                        # get the top 10 neareast neighbors wrt Jaccard
                        top10 = list(filtered_graph_sim[str(entID)].keys())[:10]
                        top10 = np.array([int(entID) for entID in top10])
                        # get the top 10 nearest neighbors wrt embedding
                        top10_emb = (
                            topk_indices[entID].numpy()[:10]
                            if dataset_name != "FB15K-237" or model_name == "rdf2vec"
                            else topk_indices[myIDs2contiguous[entID]].numpy()[:10]
                        )
                        # Perform the row-wise comparison
                        rbo_class10 = rbo_score(top10, top10_emb)
                        rbo_dict[model_name][str(class_name)]["rbo10"] += rbo_class10

                    if compute_rbo5 and number_non_zero >= 5:
                        cpt_rbo5 += 1
                        # get the top 5 neareast neighbors wrt Jaccard
                        top5 = list(filtered_graph_sim[str(entID)].keys())[:5]
                        top5 = np.array([int(entID) for entID in top5])
                        # get the top 5 nearest neighbors wrt embedding
                        top5_emb = (
                            topk_indices[entID].numpy()[:5]
                            if dataset_name != "FB15K-237" or model_name == "rdf2vec"
                            else topk_indices[myIDs2contiguous[entID]].numpy()[:5]
                        )
                        # Perform the row-wise comparison
                        rbo_class5 = rbo_score(top5, top5_emb)
                        rbo_dict[model_name][str(class_name)]["rbo5"] += rbo_class5

                    if compute_rbo3 and number_non_zero >= 3:
                        cpt_rbo3 += 1
                        top3 = list(filtered_graph_sim[str(entID)].keys())[:3]
                        top3 = np.array([int(entID) for entID in top3])
                        # get the top 3 nearest neighbors wrt embedding
                        top3_emb = (
                            topk_indices[entID].numpy()[:3]
                            if dataset_name != "FB15K-237" or model_name == "rdf2vec"
                            else topk_indices[myIDs2contiguous[entID]].numpy()[:3]
                        )
                        # Perform the row-wise comparison
                        rbo_class3 = rbo_score(top3, top3_emb)
                        rbo_dict[model_name][str(class_name)]["rbo3"] += rbo_class3

                if compute_rbo100:
                    if cpt_rbo100 != 0:
                        rbo_dict[model_name][str(class_name)]["rbo100"] /= cpt_rbo100
                    else:
                        rbo_dict[model_name][str(class_name)].pop("rbo100")
                if compute_rbo50:
                    if cpt_rbo50 != 0:
                        rbo_dict[model_name][str(class_name)]["rbo50"] /= cpt_rbo50
                    else:
                        rbo_dict[model_name][str(class_name)].pop("rbo50")
                if compute_rbo10:
                    if cpt_rbo10 != 0:
                        rbo_dict[model_name][str(class_name)]["rbo10"] /= cpt_rbo10
                    else:
                        rbo_dict[model_name][str(class_name)].pop("rbo10")
                if compute_rbo5:
                    if cpt_rbo5 != 0:
                        rbo_dict[model_name][str(class_name)]["rbo5"] /= cpt_rbo5
                    else:
                        rbo_dict[model_name][str(class_name)].pop("rbo5")
                if compute_rbo3:
                    if cpt_rbo3 != 0:
                        rbo_dict[model_name][str(class_name)]["rbo3"] /= cpt_rbo3
                    else:
                        rbo_dict[model_name][str(class_name)].pop("rbo3")

                current_time = time.time()
                if current_time - start_time >= interval:
                    print(
                        f"Currently processing: Model: {model_name}, Class: {class_name}"
                    )
                    start_time = time.time()  # Reset the timer

            # Save the results
            with open(filename, "w") as json_file:
                json.dump(rbo_dict, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Arguments with defaults
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        default="YAGO4-19K",
        help="Name of the dataset",
    )
    parser.add_argument("-s", "--split", type=str, default="train", help="train/full")
    parser.add_argument("-k", "--topk", type=int, default=100, help="Top K")
    parser.add_argument(
        "-ho", "--hops", type=int, default=0, help="1-hop (1) or 2-hops (2) or both (0)"
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="preprocessed_datasets",
        help="Path to preprocessed datasets",
    )
    parser.add_argument(
        "-sp",
        "--save_path",
        type=str,
        default="preprocessed_datasets",
        help="Path to save output file",
    )
    parser.add_argument("-v", "--verbose", type=bool, default=True, help="Verbose mode")
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=60,
        help="Interval (seconds) for printing the current model and class",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call main function
    main(args, MODELS)
