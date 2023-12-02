from utils import *

import json
import pickle
import torch
from typing import List
import numpy as np
import pykeen.nn
import random
import os
import argparse
from tqdm import tqdm
import time

# Get the current timestamp as a string
timestamp = str(int(time.time()))

DATASET_DIR = "preprocessed_datasets"
MODELS = [
    "boxe",
    "conve",
    "distmult",
    "rdf2vec",
    "rescal",
    "transd",
    "transe",
    "tucker",
]

# some classes if args.class_name is not specified
DATASET_CLASSES = {
    "FB15K-237": ["award", "tv", "people", "film", "music", "education", "medicine"],
    "AIFB": [
        "<http://swrc.ontoware.org/ontology#Publication>",
        "<http://swrc.ontoware.org/ontology#TechnicalReport>",
        "<http://swrc.ontoware.org/ontology#ResearchTopic>",
        "<http://swrc.ontoware.org/ontology#Article>",
        "<http://swrc.ontoware.org/ontology#InProceedings>",
        "<http://swrc.ontoware.org/ontology#Person>",
        "<http://swrc.ontoware.org/ontology#Project>",
        "<http://swrc.ontoware.org/ontology#Organization>",
        "<http://swrc.ontoware.org/ontology#Book>",
    ],
    "YAGO4-19K": [
        "http://schema.org/CreativeWork",
        "http://schema.org/City",
        "http://schema.org/Country",
        "http://schema.org/Product",
        "http://schema.org/TVSeries",
        "http://schema.org/Person",
    ],
}


def main(args):
    dataset_name = args.dataset_name
    dataset_path = os.path.join(DATASET_DIR, dataset_name)
    K = args.topk
    split_type = args.split
    cl = args.class_name
    cutoff = args.cutoff

    # assert that K is either 3, 5, or 10. If not, set at 10
    if K not in [3, 5, 10]:
        K = 10

    class2id = load_json(dataset_path, f"class2id")
    id2class = {v: k for k, v in class2id.items()}

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

    print("Number of classes in class2id: ", len(class2id))
    print("Number of classes in class2ent: ", len(class2ent))
    if len(class2id) != len(class2ent):
        print("Warning: class2id and class2ent do not have the same number of classes")

    if cl == "random":
        cl = random.choice(DATASET_CLASSES[dataset_name])
        print(f"Randomly selected class: {cl}")

    else:
        if dataset_name == "YAGO4-19K":
            if "http://schema.org/" + cl in class2id:
                cl = "http://schema.org/" + cl
            else:
                cl = "http://yago-knowledge.org/resource/" + cl

        if dataset_name == "AIFB":
            cl = "<http://swrc.ontoware.org/ontology#" + cl + ">"

        if dataset_name in ["db50_trans", "db50_ind"]:
            cl = "<http://dbpedia.org/ontology/" + cl + ">"

    # Define the filename with the timestamp
    stripped_cl = cl.split("/")[-1] if dataset_name == "YAGO4-19K" else cl
    stripped_cl = cl.split("#")[-1][:-1] if dataset_name == "AIFB" else stripped_cl
    stripped_cl = (
        cl.split("/")[-1][:-1]
        if dataset_name in ["db50_trans", "db50_ind"]
        else stripped_cl
    )
    stripped_cl = (
        stripped_cl.replace("/", "-") if dataset_name == "FB15K-237" else stripped_cl
    )
    filename = (
        f"logs/{dataset_name}/expe3/{timestamp}_first_hop_{stripped_cl}_K={K}.txt"
    )
    if not os.path.exists(f"logs/{dataset_name}/expe3"):
        os.makedirs(f"logs/{dataset_name}/expe3")

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
        entity2id = {}

        file_path = os.path.join(dataset_path, "id2name.txt")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    id, name = line.strip().split("\t")
                    id2name[int(id)] = name
                    entity2id[name] = int(id)
        else:
            print(f"Warning: {file_path} does not exist.")
    else:
        entity2id = read_entity2id(dataset_path)
        id2name = {int(id): name for name, id in entity2id.items()}

    if dataset_name == "FB15K-237":
        entity2id = read_entity2id(dataset_path)
        id2name = {int(id): entity for entity, id in entity2id.items()}

    if cutoff == 0:
        cutoff = len(entity2id.keys()) if dataset_name != "FB15K-237" else 14505

    relation2id = read_relation2id(dataset_path)
    id2relation = {int(id): relation for relation, id in relation2id.items()}

    rel2inv = {"inv_" + rel: id + len(relation2id) for rel, id in relation2id.items()}
    id2rel_inv = {id: rel for rel, id in rel2inv.items()}
    # extend the dicts with the inverse relations
    relation2id.update(rel2inv)
    id2relation.update(id2rel_inv)

    subgraphs_1_hop = get_all_1_hops(kg, dummy=True)
    subgraphs_both_hops = get_both_hops_optimized(kg, dummy=True)
    # remove from subgraphs_2_hops the triples that are already in subgraphs_1_hop
    subgraphs_2_hops = {
        key: value[len(subgraphs_1_hop[key]) :]
        for key, value in subgraphs_both_hops.items()
    }

    meta_dict = {}
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
                    pykeen2id = parse_pykeen_dict(
                        f"embeddings/{dataset_name}/transe/training_triples/entity_to_id.tsv.gz",
                        format="dict",
                    )
                    id2pykeen = {v: k for k, v in pykeen2id.items()}
                    if dataset_name == "FB15K-237":
                        id2pykeen = dict(sorted(id2pykeen.items()))
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

        # save topk_indices on disk if it does not exist and load it
        if not os.path.exists(
            os.path.join(
                dataset_path,
                f"topk_indices_{split_type}_model={model_name}_hops=0_cutoff={cutoff}.pt",
            )
        ):
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
            topk_values, topk_indices = torch.topk(similarity_matrix, k=cutoff, dim=1)
            torch.save(
                topk_indices.clone(),
                os.path.join(
                    dataset_path,
                    f"topk_indices_{split_type}_model={model_name}_hops=0_cutoff={cutoff}.pt",
                ),
            )
            print(f"Saved topk_indices on disk for {model_name}")
        else:
            topk_indices = torch.load(
                os.path.join(
                    dataset_path,
                    f"topk_indices_{split_type}_model={model_name}_hops=0_cutoff={cutoff}.pt",
                )
            )
            print(f"Loaded topk_indices from disk for {model_name}")

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
            dataset_path, f"fast_jaccard2id_{split_type}_K=100_hops=0"
        )

        filtered_graph_sim = {
            entID: inner_dict
            for entID, inner_dict in graph_sim.items()
            if all(jaccard_val != 0.0 for jaccard_val in list(inner_dict.values())[:K])
        }
        print(
            f"Keeping only entities with >= {K} non-zero Jaccard coefficients. Remaining number of entities: {len(filtered_graph_sim)}"
        )
        # order the keys
        filtered_graph_sim = dict(
            sorted(filtered_graph_sim.items(), key=lambda x: int(x[0]))
        )

        if model_name != "rdf2vec":
            to_keep = [int(entID) for entID in filtered_graph_sim.keys()]

            if dataset_name == "FB15K-237":
                to_keep_original = to_keep.copy()
                to_keep = [myIDs2contiguous[entID] for entID in to_keep]

        else:
            if dataset_name == "FB15K-237":
                to_keep_original = [int(entID) for entID in filtered_graph_sim.keys()]
                # map the keys and values using id_original2id_rdf2vec
                filtered_graph_sim_copy = filtered_graph_sim.copy()
                filtered_graph_sim = {}
                for key, value in filtered_graph_sim_copy.items():
                    filtered_graph_sim[str(id_original2id_rdf2vec[int(key)])] = {
                        id_original2id_rdf2vec[int(k)]: v for k, v in value.items()
                    }

            # Create a PyTorch tensor from the list of keys
            to_keep = [int(entID) for entID in filtered_graph_sim.keys()]

        classID = class2id[cl]

        entitiesID = class2ent2id[str(classID)]
        print(f"Number of entities of class {stripped_cl}:", len(entitiesID))

        if model_name == "rdf2vec":
            if dataset_name == "FB15K-237":
                entitiesID = [
                    int(entID) for entID in entitiesID if int(entID) in to_keep_original
                ]
                entitiesID = [id_original2id_rdf2vec[entID] for entID in entitiesID]
            else:
                entitiesID = [
                    int(entID) for entID in entitiesID if int(entID) in to_keep
                ]

        entitiesID = [int(entID) for entID in entitiesID if int(entID) in to_keep]
        print(
            f"Number of entities of class {stripped_cl} with >= {K} non-zero Jaccard coefficients:",
            len(entitiesID),
        )

        total_triples_1_hop = {}
        (
            total_triples_1_hop["top10"],
            total_triples_1_hop["top5"],
            total_triples_1_hop["top3"],
        ) = ({}, {}, {})
        (
            total_triples_1_hop["top10_filtered"],
            total_triples_1_hop["top5_filtered"],
            total_triples_1_hop["top3_filtered"],
        ) = ({}, {}, {})

        total_intersections_1_hop = {}
        (
            total_intersections_1_hop["top10"],
            total_intersections_1_hop["top5"],
            total_intersections_1_hop["top3"],
        ) = ({}, {}, {})
        (
            total_intersections_1_hop["top10_filtered"],
            total_intersections_1_hop["top5_filtered"],
            total_intersections_1_hop["top3_filtered"],
        ) = ({}, {}, {})

        counter_predicates_1_hop = {}
        (
            counter_predicates_1_hop["top10"],
            counter_predicates_1_hop["top5"],
            counter_predicates_1_hop["top3"],
        ) = ({}, {}, {})
        (
            counter_predicates_1_hop["top10_filtered"],
            counter_predicates_1_hop["top5_filtered"],
            counter_predicates_1_hop["top3_filtered"],
        ) = ({}, {}, {})

        weights_predicates_1_hop = {}
        (
            weights_predicates_1_hop["top10"],
            weights_predicates_1_hop["top5"],
            weights_predicates_1_hop["top3"],
        ) = ({}, {}, {})
        (
            weights_predicates_1_hop["top10_filtered"],
            weights_predicates_1_hop["top5_filtered"],
            weights_predicates_1_hop["top3_filtered"],
        ) = ({}, {}, {})

        # ----------------------------------------#

        total_triples_first_hop = {}
        (
            total_triples_first_hop["top10"],
            total_triples_first_hop["top5"],
            total_triples_first_hop["top3"],
        ) = ({}, {}, {})
        (
            total_triples_first_hop["top10_filtered"],
            total_triples_first_hop["top5_filtered"],
            total_triples_first_hop["top3_filtered"],
        ) = ({}, {}, {})

        total_intersections_first_hop = {}
        (
            total_intersections_first_hop["top10"],
            total_intersections_first_hop["top5"],
            total_intersections_first_hop["top3"],
        ) = ({}, {}, {})
        (
            total_intersections_first_hop["top10_filtered"],
            total_intersections_first_hop["top5_filtered"],
            total_intersections_first_hop["top3_filtered"],
        ) = ({}, {}, {})

        counter_predicates_first_hop = {}
        (
            counter_predicates_first_hop["top10"],
            counter_predicates_first_hop["top5"],
            counter_predicates_first_hop["top3"],
        ) = ({}, {}, {})
        (
            counter_predicates_first_hop["top10_filtered"],
            counter_predicates_first_hop["top5_filtered"],
            counter_predicates_first_hop["top3_filtered"],
        ) = ({}, {}, {})

        weights_predicates_first_hop = {}
        (
            weights_predicates_first_hop["top10"],
            weights_predicates_first_hop["top5"],
            weights_predicates_first_hop["top3"],
        ) = ({}, {}, {})
        (
            weights_predicates_first_hop["top10_filtered"],
            weights_predicates_first_hop["top5_filtered"],
            weights_predicates_first_hop["top3_filtered"],
        ) = ({}, {}, {})

        # ----------------------------------------#

        total_triples_both_hop = {}
        (
            total_triples_both_hop["top10"],
            total_triples_both_hop["top5"],
            total_triples_both_hop["top3"],
        ) = ({}, {}, {})
        (
            total_triples_both_hop["top10_filtered"],
            total_triples_both_hop["top5_filtered"],
            total_triples_both_hop["top3_filtered"],
        ) = ({}, {}, {})

        total_intersections_both_hop = {}
        (
            total_intersections_both_hop["top10"],
            total_intersections_both_hop["top5"],
            total_intersections_both_hop["top3"],
        ) = ({}, {}, {})
        (
            total_intersections_both_hop["top10_filtered"],
            total_intersections_both_hop["top5_filtered"],
            total_intersections_both_hop["top3_filtered"],
        ) = ({}, {}, {})

        counter_predicates_both_hop = {}
        (
            counter_predicates_both_hop["top10"],
            counter_predicates_both_hop["top5"],
            counter_predicates_both_hop["top3"],
        ) = ({}, {}, {})
        (
            counter_predicates_both_hop["top10_filtered"],
            counter_predicates_both_hop["top5_filtered"],
            counter_predicates_both_hop["top3_filtered"],
        ) = ({}, {}, {})

        weights_predicates_both_hop = {}
        (
            weights_predicates_both_hop["top10"],
            weights_predicates_both_hop["top5"],
            weights_predicates_both_hop["top3"],
        ) = ({}, {}, {})
        (
            weights_predicates_both_hop["top10_filtered"],
            weights_predicates_both_hop["top5_filtered"],
            weights_predicates_both_hop["top3_filtered"],
        ) = ({}, {}, {})

        two_hops_paths = {}

        for entID in tqdm(entitiesID):
            subgraph_1_hop_current_entity = subgraphs_1_hop[entID]
            subgraph_2_hop_current_entity = subgraphs_2_hops[entID]

            # order the full list of neighbors by their embedding-based similarity
            neighbors = (
                topk_indices[entID].numpy()
                if dataset_name != "FB15K-237" or model_name == "rdf2vec"
                else topk_indices[myIDs2contiguous[entID]].numpy()
            )
            # print("len neighbors: ", len(neighbors))
            neighbors_top10, neighbors_top5, neighbors_top3 = (
                neighbors[:10],
                neighbors[:5],
                neighbors[:3],
            )
            filtered_neighbors = [
                neighbor
                for neighbor in neighbors
                if neighbor in class2ent2id[str(classID)]
            ]
            # print("len filtered neighbors: ", len(filtered_neighbors))
            (
                filtered_neighbors_top10,
                filtered_neighbors_top5,
                filtered_neighbors_top3,
            ) = (
                filtered_neighbors[:10],
                filtered_neighbors[:5],
                filtered_neighbors[:3],
            )

            unique_predicates_entID_1_hop = set(subgraph_1_hop_current_entity[:, 1])
            # unique_predicates_entID_first_hop = set(subgraph_2_hop_current_entity[:, 1]) # HARD TO KNOW: if there are cycles, which one is the first predicate?
            unique_predicates_entID_both_hop = subgraph_2_hop_current_entity[:, [0, 1]]
            unique_predicates_entID_both_hop = set(
                map(tuple, unique_predicates_entID_both_hop)
            )

            # FILTERED (same class only)
            for neigh in filtered_neighbors_top10:
                subgraph_1_hop_neighbor = subgraphs_1_hop[neigh]
                subgraph_2_hop_neighbor = subgraphs_2_hops[neigh]
                in_top3 = neigh in filtered_neighbors_top3
                in_top5 = neigh in filtered_neighbors_top5

                # 1-HOP
                unique_predicates_neighbor_1_hop = set(subgraph_1_hop_neighbor[:, 1])
                shared_predicates_1_hop = (
                    unique_predicates_entID_1_hop & unique_predicates_neighbor_1_hop
                )
                # below should be the number of common triples in the 1-hop subgraphs, so not only same predicate but same everything
                intersecting_triples_1_hop = set(
                    map(tuple, subgraph_1_hop_current_entity)
                ) & set(map(tuple, subgraph_1_hop_neighbor))
                num_intersecting_triples_1_hop = len(intersecting_triples_1_hop)
                intersecting_triples_1_hop = np.array(list(intersecting_triples_1_hop))

                if len(intersecting_triples_1_hop) > 0:
                    # weight each predicate by its proportion over the total number of intersecting triples
                    for predicate in shared_predicates_1_hop:
                        # number of intersecting triples having this predicate
                        num_intersecting_triples_1_hop_predicate = len(
                            intersecting_triples_1_hop[
                                intersecting_triples_1_hop[:, 1] == predicate
                            ]
                        )
                        counter_predicates_1_hop["top10_filtered"][
                            id2relation[predicate]
                        ] = (
                            counter_predicates_1_hop["top10_filtered"].get(
                                id2relation[predicate], 0
                            )
                            + num_intersecting_triples_1_hop_predicate
                        )
                        predicate_weight = (
                            num_intersecting_triples_1_hop_predicate
                            / num_intersecting_triples_1_hop
                        )
                        weights_predicates_1_hop["top10_filtered"][
                            id2relation[predicate]
                        ] = (
                            weights_predicates_1_hop["top10_filtered"].get(
                                id2relation[predicate], 0.0
                            )
                            + predicate_weight
                        )

                        if in_top3:
                            counter_predicates_1_hop["top3_filtered"][
                                id2relation[predicate]
                            ] = (
                                counter_predicates_1_hop["top3_filtered"].get(
                                    id2relation[predicate], 0
                                )
                                + num_intersecting_triples_1_hop_predicate
                            )
                            weights_predicates_1_hop["top3_filtered"][
                                id2relation[predicate]
                            ] = (
                                weights_predicates_1_hop["top3_filtered"].get(
                                    id2relation[predicate], 0.0
                                )
                                + predicate_weight
                            )
                        if in_top5:
                            counter_predicates_1_hop["top5_filtered"][
                                id2relation[predicate]
                            ] = (
                                counter_predicates_1_hop["top5_filtered"].get(
                                    id2relation[predicate], 0
                                )
                                + num_intersecting_triples_1_hop_predicate
                            )
                            weights_predicates_1_hop["top5_filtered"][
                                id2relation[predicate]
                            ] = (
                                weights_predicates_1_hop["top5_filtered"].get(
                                    id2relation[predicate], 0.0
                                )
                                + predicate_weight
                            )

                # 2-HOP
                unique_predicates_neighbor_both_hop = subgraph_2_hop_neighbor[:, [0, 1]]
                unique_predicates_neighbor_both_hop = set(
                    map(tuple, unique_predicates_neighbor_both_hop)
                )
                shared_predicates_both_hop = (
                    unique_predicates_entID_both_hop
                    & unique_predicates_neighbor_both_hop
                )
                intersecting_triples_both_hop = set(
                    map(tuple, subgraph_2_hop_current_entity)
                ) & set(map(tuple, subgraph_2_hop_neighbor))
                num_intersecting_triples_both_hop = len(intersecting_triples_both_hop)
                intersecting_triples_both_hop = np.array(
                    list(intersecting_triples_both_hop)
                )

                if len(intersecting_triples_both_hop) > 0:
                    # weight each predicate by its proportion over the total number of intersecting triples
                    for predicates in shared_predicates_both_hop:
                        # number of intersecting triples having this predicate
                        predicate1, predicate2 = predicates[0], predicates[1]
                        intersecting_triples_both_hop_predicate = (
                            intersecting_triples_both_hop[
                                (intersecting_triples_both_hop[:, 0] == predicate1)
                                & (intersecting_triples_both_hop[:, 1] == predicate2)
                            ]
                        )
                        num_intersecting_triples_both_hop_predicate = len(
                            intersecting_triples_both_hop_predicate
                        )
                        counter_predicates_both_hop["top10_filtered"][
                            f"{id2relation[predicate1]}, {id2relation[predicate2]}"
                        ] = (
                            counter_predicates_both_hop["top10_filtered"].get(
                                f"{id2relation[predicate1]}, {id2relation[predicate2]}",
                                0,
                            )
                            + num_intersecting_triples_both_hop_predicate
                        )
                        predicate_weight = (
                            num_intersecting_triples_both_hop_predicate
                            / num_intersecting_triples_both_hop
                        )
                        weights_predicates_both_hop["top10_filtered"][
                            f"{id2relation[predicate1]}, {id2relation[predicate2]}"
                        ] = (
                            weights_predicates_both_hop["top10_filtered"].get(
                                f"{id2relation[predicate1]}, {id2relation[predicate2]}",
                                0.0,
                            )
                            + predicate_weight
                        )

                        if in_top3:
                            counter_predicates_both_hop["top3_filtered"][
                                f"{id2relation[predicate1]}, {id2relation[predicate2]}"
                            ] = (
                                counter_predicates_both_hop["top3_filtered"].get(
                                    f"{id2relation[predicate1]}, {id2relation[predicate2]}",
                                    0,
                                )
                                + num_intersecting_triples_both_hop_predicate
                            )
                            weights_predicates_both_hop["top3_filtered"][
                                f"{id2relation[predicate1]}, {id2relation[predicate2]}"
                            ] = (
                                weights_predicates_both_hop["top3_filtered"].get(
                                    f"{id2relation[predicate1]}, {id2relation[predicate2]}",
                                    0.0,
                                )
                                + predicate_weight
                            )
                        if in_top5:
                            counter_predicates_both_hop["top5_filtered"][
                                f"{id2relation[predicate1]}, {id2relation[predicate2]}"
                            ] = (
                                counter_predicates_both_hop["top5_filtered"].get(
                                    f"{id2relation[predicate1]}, {id2relation[predicate2]}",
                                    0,
                                )
                                + num_intersecting_triples_both_hop_predicate
                            )
                            weights_predicates_both_hop["top5_filtered"][
                                f"{id2relation[predicate1]}, {id2relation[predicate2]}"
                            ] = (
                                weights_predicates_both_hop["top5_filtered"].get(
                                    f"{id2relation[predicate1]}, {id2relation[predicate2]}",
                                    0.0,
                                )
                                + predicate_weight
                            )

        denominator_top10 = len(entitiesID) * 10
        denominator_top5 = len(entitiesID) * 5
        denominator_top3 = len(entitiesID) * 3

        # 1-HOP
        # compute the macro average for top 10, top 5, top 3
        for predicate in weights_predicates_1_hop["top10_filtered"].keys():
            weights_predicates_1_hop["top10_filtered"][predicate] /= denominator_top10
        for predicate in weights_predicates_1_hop["top5_filtered"].keys():
            weights_predicates_1_hop["top5_filtered"][predicate] /= denominator_top5
        for predicate in weights_predicates_1_hop["top3_filtered"].keys():
            weights_predicates_1_hop["top3_filtered"][predicate] /= denominator_top3

        # create a normalized version
        # 1-HOP
        weights_predicates_1_hop["top10_filtered_normalized"] = {}
        weights_predicates_1_hop["top5_filtered_normalized"] = {}
        weights_predicates_1_hop["top3_filtered_normalized"] = {}

        for predicate in weights_predicates_1_hop["top10_filtered"].keys():
            weights_predicates_1_hop["top10_filtered_normalized"][
                predicate
            ] = counter_predicates_1_hop["top10_filtered"][predicate] / sum(
                counter_predicates_1_hop["top10_filtered"].values()
            )
        for predicate in weights_predicates_1_hop["top5_filtered"].keys():
            weights_predicates_1_hop["top5_filtered_normalized"][
                predicate
            ] = counter_predicates_1_hop["top5_filtered"][predicate] / sum(
                counter_predicates_1_hop["top5_filtered"].values()
            )
        for predicate in weights_predicates_1_hop["top3_filtered"].keys():
            weights_predicates_1_hop["top3_filtered_normalized"][
                predicate
            ] = counter_predicates_1_hop["top3_filtered"][predicate] / sum(
                counter_predicates_1_hop["top3_filtered"].values()
            )

        # 2-HOP
        for predicate in weights_predicates_both_hop["top10_filtered"].keys():
            weights_predicates_both_hop["top10_filtered"][
                predicate
            ] /= denominator_top10
        for predicate in weights_predicates_both_hop["top5_filtered"].keys():
            weights_predicates_both_hop["top5_filtered"][predicate] /= denominator_top5
        for predicate in weights_predicates_both_hop["top3_filtered"].keys():
            weights_predicates_both_hop["top3_filtered"][predicate] /= denominator_top3

        # 1-HOP
        weights_predicates_1_hop["top10_filtered"] = dict(
            sorted(
                weights_predicates_1_hop["top10_filtered"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        weights_predicates_1_hop["top5_filtered"] = dict(
            sorted(
                weights_predicates_1_hop["top5_filtered"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        weights_predicates_1_hop["top3_filtered"] = dict(
            sorted(
                weights_predicates_1_hop["top3_filtered"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        weights_predicates_1_hop["top10_filtered_normalized"] = dict(
            sorted(
                weights_predicates_1_hop["top10_filtered_normalized"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        weights_predicates_1_hop["top5_filtered_normalized"] = dict(
            sorted(
                weights_predicates_1_hop["top5_filtered_normalized"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        weights_predicates_1_hop["top3_filtered_normalized"] = dict(
            sorted(
                weights_predicates_1_hop["top3_filtered_normalized"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        counter_predicates_1_hop["top10_filtered"] = dict(
            sorted(
                counter_predicates_1_hop["top10_filtered"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        counter_predicates_1_hop["top5_filtered"] = dict(
            sorted(
                counter_predicates_1_hop["top5_filtered"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        counter_predicates_1_hop["top3_filtered"] = dict(
            sorted(
                counter_predicates_1_hop["top3_filtered"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        # 2-HOP
        weights_predicates_both_hop["top10_filtered"] = dict(
            sorted(
                weights_predicates_both_hop["top10_filtered"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        weights_predicates_both_hop["top5_filtered"] = dict(
            sorted(
                weights_predicates_both_hop["top5_filtered"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        weights_predicates_both_hop["top3_filtered"] = dict(
            sorted(
                weights_predicates_both_hop["top3_filtered"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        counter_predicates_both_hop["top10_filtered"] = dict(
            sorted(
                counter_predicates_both_hop["top10_filtered"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        counter_predicates_both_hop["top5_filtered"] = dict(
            sorted(
                counter_predicates_both_hop["top5_filtered"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        counter_predicates_both_hop["top3_filtered"] = dict(
            sorted(
                counter_predicates_both_hop["top3_filtered"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        meta_dict[model_name] = {}
        meta_dict[model_name]["counter_predicates_1_hop"] = counter_predicates_1_hop
        meta_dict[model_name]["weights_predicates_1_hop"] = weights_predicates_1_hop

    with open(f"logs/{dataset_name}/expe3/1_hop_{stripped_cl}.json", "w") as file:
        json.dump(meta_dict, file, indent=4)


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
    parser.add_argument(
        "-c",
        "--class_name",
        type=str,
        default="Painting",
        help="Class on which sample entities. Either class_label or random",
    )
    parser.add_argument("-k", "--topk", type=int, default=10, help="Top K")
    parser.add_argument(
        "-cu",
        "--cutoff",
        type=int,
        default=0,
        help="Cutoff for keeping the first K rows for each entry of the similarity matrix. If 0, keep all rows",
    )
    parser.add_argument(
        "-l", "--load", type=int, default=1, help="Load similarity matrix from disk"
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

    # Parse the arguments
    args = parser.parse_args()

    # Call main function
    main(args)
