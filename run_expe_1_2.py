from utils import *

import pickle
import json
from scipy.stats import pearsonr
import torch
from typing import List
import numpy as np
import pykeen.nn
import sys
import os
import argparse
from tqdm import tqdm
import time

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


def main(args):
    dataset_name = args.dataset_name
    dataset_path = os.path.join(DATASET_DIR, dataset_name)
    K = args.topk
    R = args.ranked
    min_entities = args.min_entities
    filter = args.filter
    hops = args.hops
    split_type = args.split

    # Initialize lists to store extracted data
    models = []
    mrr_values = []
    hits10_values = []
    hits3_values = []
    rbo100_values = []
    rbo10_values = []
    rbo3_values = []

    # Define the filename with the timestamp
    filename = f"logs/{dataset_name}/expe_1_and_2/{timestamp}_K={K}_R={R}_min_ent={min_entities}_filter={filter}_hops={hops}.txt"
    if not os.path.exists(f"logs/{dataset_name}/expe_1_and_2"):
        os.makedirs(f"logs/{dataset_name}/expe_1_and_2")

    with open(filename, "a") as file:
        file.write(50 * "-" + "\n")
        file.write("Timestamp: " + timestamp + "\n")
        file.write("Dataset: " + dataset_name + "\n")
        file.write("Split type: " + split_type + "\n")
        file.write("Models: " + str(MODELS) + "\n")
        file.write("Top K: " + str(K) + "\n")
        file.write("Min entities per class: " + str(min_entities) + "\n")
        file.write("Filter: " + str(filter) + "\n")
        if hops == 0:
            file.write("Hops: 1 and 2\n")
        else:
            file.write("Hops: " + str(hops) + "\n")
        file.write(50 * "-" + "\n")

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
                entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[
                    0
                ]
                if model_name == "transd" and dataset_name == "FB15K-237":
                    entity_embedding_tensor = (
                        entity_embeddings._embeddings.weight.cpu().detach()
                    )
                else:
                    entity_embedding_tensor: torch.FloatTensor = (
                        entity_embeddings().cpu().detach()
                    )
                # Only for models trained with Pykeen (i.e. != RDF2Vec)
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
                    print(f"Model {model_name} not found for dataset {dataset_name}")
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
                os.path.join("embeddings", dataset_name, "rdf2vec", "relations.txt"),
                sep="\t",
                header=None,
            )
            entities2id_rdf2vec = {entity: id for id, entity in enumerate(entities[0])}
            relations2id_rdf2vec = {
                relation: id for id, relation in enumerate(relations[0])
            }
            assert entity2id == entities2id_rdf2vec
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

        # keep only items for which none of the top K jaccard is 0 (as after Jaccard of 0, ranking is arbitrary)
        filtered_graph_sim = {
            entID: inner_dict
            for entID, inner_dict in graph_sim.items()
            if all(jaccard_val != 0.0 for jaccard_val in inner_dict.values())
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

        np_topk_indices_graph_based, np_topk_indices_embedding_based = (
            topk_indices_graph_based.numpy(),
            topk_indices_embedding_based.numpy(),
        )

        # Apply rbo_score row-wise and store the results in a list
        rbo_scores = [
            rbo_score(row1, row2)
            for row1, row2 in zip(
                np_topk_indices_graph_based, np_topk_indices_embedding_based
            )
        ]

        # Convert the list of scores to a NumPy array
        rbo_scores_array = np.array(rbo_scores).reshape(-1, 1)

        rbo100 = round(rbo_scores_array.mean(), 3)

        # RBO per class
        class2rbo100 = {}
        class2rbo10 = {}
        class2rbo3 = {}
        for classID in class2id.values():
            compute_rbo3, compute_rbo10, compute_rbo100 = True, True, True
            # Get the entities for this class
            entitiesID = class2ent2id[str(classID)]
            if dataset_name == "FB15K-237":
                entitiesID = [
                    int(entID) for entID in entitiesID if int(entID) in to_keep_original
                ]
                if model_name == "rdf2vec":
                    entitiesID = [id_original2id_rdf2vec[entID] for entID in entitiesID]
            else:
                entitiesID = [
                    int(entID) for entID in entitiesID if int(entID) in to_keep
                ]
            # pass to the next class if there are no entities for this class
            if len(entitiesID) == 0:
                continue
            # activate RBO computation only if there are at least min_entities entities for this class
            if filter == 1:
                if len(entitiesID) >= 100:
                    pass
                elif len(entitiesID) >= 10:
                    compute_rbo100 = False
                elif len(entitiesID) >= 3:
                    compute_rbo100 = False
                    compute_rbo10 = False
                else:
                    compute_rbo3 = False
                    compute_rbo10 = False
                    compute_rbo100 = False
                    # pass to the next class
                    continue

            common_elements_count = []
            rbo_class100 = 0.0
            rbo_class10 = 0.0
            rbo_class3 = 0.0
            if len(entitiesID) >= min_entities:
                for entID in entitiesID:
                    if compute_rbo100:
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
                        rbo_class100 += rbo_score(top100, top100_emb)
                    if compute_rbo10:
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
                        rbo_class10 += rbo_score(top10, top10_emb)
                    if compute_rbo3:
                        top3 = list(filtered_graph_sim[str(entID)].keys())[:3]
                        top3 = np.array([int(entID) for entID in top3])
                        # get the top 3 nearest neighbors wrt embedding
                        top3_emb = (
                            topk_indices[entID].numpy()[:3]
                            if dataset_name != "FB15K-237" or model_name == "rdf2vec"
                            else topk_indices[myIDs2contiguous[entID]].numpy()[:3]
                        )
                        # Perform the row-wise comparison
                        rbo_class3 += rbo_score(top3, top3_emb)

                if compute_rbo3:
                    class2rbo3[id2class[classID]] = {
                        "num_entities": len(entitiesID),
                        "rbo": round(rbo_class3 / len(entitiesID), 3),
                    }
                if compute_rbo10:
                    class2rbo10[id2class[classID]] = {
                        "num_entities": len(entitiesID),
                        "rbo": round(rbo_class10 / len(entitiesID), 3),
                    }
                if compute_rbo100:
                    class2rbo100[id2class[classID]] = {
                        "num_entities": len(entitiesID),
                        "rbo": round(rbo_class100 / len(entitiesID), 3),
                    }

        ##########################################################
        # K = 10
        topk_indices_embedding_based = topk_indices_embedding_based[:, np.arange(10)]
        topk_indices_graph_based = topk_indices_graph_based[:, np.arange(10)]

        # Perform the row-wise comparison
        common_elements_count = np.sum(
            np.array(topk_indices_graph_based)[:, np.newaxis, :]
            == np.array(topk_indices_embedding_based)[:, :, np.newaxis],
            axis=2,
        )

        # Sum the common elements count row-wise
        sum_common_elements = np.sum(common_elements_count, axis=1)

        np_topk_indices_graph_based, np_topk_indices_embedding_based = (
            topk_indices_graph_based.numpy(),
            topk_indices_embedding_based.numpy(),
        )

        # Apply rbo_score row-wise and store the results in a list
        rbo_scores = [
            rbo_score(row1, row2)
            for row1, row2 in zip(
                np_topk_indices_graph_based, np_topk_indices_embedding_based
            )
        ]

        # Convert the list of scores to a NumPy array of shape (1600, 1)
        rbo_scores_array = np.array(rbo_scores).reshape(-1, 1)

        rbo10 = round(rbo_scores_array.mean(), 3)

        ##########################################################
        # K = 3
        topk_indices_embedding_based = topk_indices_embedding_based[:, np.arange(3)]
        topk_indices_graph_based = topk_indices_graph_based[:, np.arange(3)]

        # Perform the row-wise comparison
        common_elements_count = np.sum(
            np.array(topk_indices_graph_based)[:, np.newaxis, :]
            == np.array(topk_indices_embedding_based)[:, :, np.newaxis],
            axis=2,
        )

        # Sum the common elements count row-wise
        sum_common_elements = np.sum(common_elements_count, axis=1)

        np_topk_indices_graph_based, np_topk_indices_embedding_based = (
            topk_indices_graph_based.numpy(),
            topk_indices_embedding_based.numpy(),
        )

        # Apply rbo_score row-wise and store the results in a list
        rbo_scores = [
            rbo_score(row1, row2)
            for row1, row2 in zip(
                np_topk_indices_graph_based, np_topk_indices_embedding_based
            )
        ]

        # Convert the list of scores to a NumPy array of shape (1600, 1)
        rbo_scores_array = np.array(rbo_scores).reshape(-1, 1)

        rbo3 = round(rbo_scores_array.mean(), 3)

        # --------------------------------------------------------#

        # load results
        if model_name != "rdf2vec":
            with open(
                f"embeddings/{dataset_name}/{model_name}/results.json", "r"
            ) as file:
                results = json.load(file)

            mrr = results["metrics"]["both"]["realistic"]["inverse_harmonic_mean_rank"]
            h1 = results["metrics"]["both"]["realistic"]["hits_at_1"]
            h3 = results["metrics"]["both"]["realistic"]["hits_at_3"]
            h10 = results["metrics"]["both"]["realistic"]["hits_at_10"]

        with open(filename, "a") as file:
            file.write("Model: " + model_name + "\n")
            if model_name != "rdf2vec":
                file.write("MRR: " + str(round(mrr, 3)) + "\n")
                file.write("Hits@1: " + str(round(h1, 3)) + "\n")
                file.write("Hits@3: " + str(round(h3, 3)) + "\n")
                file.write("Hits@10: " + str(round(h10, 3)) + "\n")
            file.write("RBO@3: " + str(rbo3) + "\n")
            file.write("RBO@10: " + str(rbo10) + "\n")
            file.write("RBO@100: " + str(rbo100) + "\n")
            file.write(
                "RBO@3 (Filt.) "
                + str(
                    round(
                        sum(
                            [
                                values["rbo"] * values["num_entities"]
                                for values in class2rbo3.values()
                            ]
                        )
                        / sum(
                            [values["num_entities"] for values in class2rbo3.values()]
                        ),
                        3,
                    )
                )
                + "\n"
            )
            file.write(
                "RBO@10 (Filt.) "
                + str(
                    round(
                        sum(
                            [
                                values["rbo"] * values["num_entities"]
                                for values in class2rbo10.values()
                            ]
                        )
                        / sum(
                            [values["num_entities"] for values in class2rbo10.values()]
                        ),
                        3,
                    )
                )
                + "\n"
            )
            file.write(
                "RBO@100 (Filt.) "
                + str(
                    round(
                        sum(
                            [
                                values["rbo"] * values["num_entities"]
                                for values in class2rbo100.values()
                            ]
                        )
                        / sum(
                            [values["num_entities"] for values in class2rbo100.values()]
                        ),
                        3,
                    )
                )
                + "\n"
            )
            file.write(f"Top {R} classes with highest RBO@3:\n")
            for cl, values in sorted(
                class2rbo3.items(), key=lambda x: x[1]["rbo"], reverse=True
            )[:R]:
                file.write(f"{cl}: {values}\n")
            file.write("\n")
            file.write(f"Top {R} classes with highest RBO@10:\n")
            for cl, values in sorted(
                class2rbo10.items(), key=lambda x: x[1]["rbo"], reverse=True
            )[:R]:
                file.write(f"{cl}: {values}\n")
            file.write("\n")
            file.write(f"Top {R} classes with highest RBO@100:\n")
            for cl, values in sorted(
                class2rbo100.items(), key=lambda x: x[1]["rbo"], reverse=True
            )[:R]:
                file.write(f"{cl}: {values}\n")
            file.write("\n")
            file.write(f"Worst {R} classes with lowest RBO@3:\n")
            for cl, values in sorted(class2rbo3.items(), key=lambda x: x[1]["rbo"])[:R]:
                file.write(f"{cl}: {values}\n")
            file.write("\n")
            file.write(f"Worst {R} classes with lowest RBO@10:\n")
            for cl, values in sorted(class2rbo10.items(), key=lambda x: x[1]["rbo"])[
                :R
            ]:
                file.write(f"{cl}: {values}\n")
            file.write("\n")
            file.write(f"Worst {R} classes with lowest RBO@100:\n")
            for cl, values in sorted(class2rbo100.items(), key=lambda x: x[1]["rbo"])[
                :R
            ]:
                file.write(f"{cl}: {values}\n")
            file.write("\n")

        if model_name != "rdf2vec":
            models.append(model_name)
            mrr_values.append(mrr)
            hits10_values.append(h10)
            hits3_values.append(h3)
            rbo100_values.append(rbo100)
            rbo10_values.append(rbo10)
            rbo3_values.append(rbo3)

    # Ensure all lists have the same length by filling missing values with NaN
    max_length = max(
        len(models),
        len(mrr_values),
        len(hits10_values),
        len(hits3_values),
        len(rbo100_values),
        len(rbo10_values),
        len(rbo3_values),
    )
    models.extend(["NaN"] * (max_length - len(models)))
    mrr_values.extend([np.nan] * (max_length - len(mrr_values)))
    hits10_values.extend([np.nan] * (max_length - len(hits10_values)))
    hits3_values.extend([np.nan] * (max_length - len(hits3_values)))
    rbo100_values.extend([np.nan] * (max_length - len(rbo100_values)))
    rbo10_values.extend([np.nan] * (max_length - len(rbo10_values)))
    rbo3_values.extend([np.nan] * (max_length - len(rbo3_values)))

    # Create a DataFrame to store the extracted data
    data = pd.DataFrame(
        {
            "Model": models,
            "MRR": mrr_values,
            "Hits@10": hits10_values,
            "Hits@3": hits3_values,
            "RBO@100": rbo100_values,
            "RBO@10": rbo10_values,
            "RBO@3": rbo3_values,
        }
    )

    # Compute correlations
    correlation_rbo100_mrr = pearsonr(data["RBO@100"], data["MRR"])[0]
    correlation_rbo10_mrr = pearsonr(data["RBO@10"], data["MRR"])[0]
    correlation_rbo3_mrr = pearsonr(data["RBO@3"], data["MRR"])[0]
    correlation_rbo10_hits10 = pearsonr(data["RBO@10"], data["Hits@10"])[0]
    correlation_rbo3_hits3 = pearsonr(data["RBO@3"], data["Hits@3"])[0]

    # Append the correlations to the file
    with open(filename, "a") as file:
        file.write(f"\nCorrelation RBO@100 and MRR: {correlation_rbo100_mrr}\n")
        file.write(f"Correlation RBO@10 and MRR: {correlation_rbo10_mrr}\n")
        file.write(f"Correlation RBO@3 and MRR: {correlation_rbo3_mrr}\n")
        file.write(f"Correlation RBO@10 and Hits@10: {correlation_rbo10_hits10}\n")
        file.write(f"Correlation RBO@3 and Hits@3: {correlation_rbo3_hits3}\n")


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
        "-m",
        "--min_entities",
        type=int,
        default=10,
        help="Minimum number of entities per class for computing RBO",
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=int,
        default=1,
        help="RBO@E for class C can only be computed if class C has at least E entities",
    )
    parser.add_argument(
        "-r",
        "--ranked",
        type=int,
        default=5,
        help="Top-C classes with highest/lowest RBO@K to display",
    )
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

    # Parse the arguments
    args = parser.parse_args()

    # Call main function
    main(args)
