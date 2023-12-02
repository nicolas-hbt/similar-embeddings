import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import json
from numpy import asarray, diagonal, minimum, newaxis
import re
import os
from tqdm import tqdm
from collections import Counter
from scipy import stats
import rbo


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # Handle numpy arrays
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def convert_kg_labels_to_ids(kg, entity2id, relation2id):
    # Create a new array with the same shape as kg to store the IDs
    kg_ids = np.empty(kg.shape, dtype=int)

    # Convert entities using entity2id dictionary
    kg_ids[:, 0] = np.vectorize(entity2id.get)(kg[:, 0])
    kg_ids[:, 2] = np.vectorize(entity2id.get)(kg[:, 2])

    # Convert relations using relation2id dictionary
    kg_ids[:, 1] = np.vectorize(relation2id.get)(kg[:, 1])

    return kg_ids


def quantize(matrix, bins=256):
    # Define the range of your coefficients
    min_val = -1.0
    max_val = 1.0

    # Create intervals
    intervals = np.linspace(min_val, max_val, bins)

    # Quantize the values into bin indices
    quantized = np.digitize(matrix, intervals)

    # Convert to uint8 to ensure 8-bit storage
    quantized = quantized.astype(np.uint8)

    return quantized


def mapping_kg2format(dataset_name):
    if dataset_name in ["DBpedia50", "FB15K"]:
        return "sop"
    return "spo"


def read_entity2id(dataset_folder):
    entity2id = {}
    try:
        with open(
            os.path.join(dataset_folder, "entity2id.txt"), "r", encoding="utf-8"
        ) as entity_dict:
            first_line = entity_dict.readline().strip()
            elements = first_line.split()

            if len(elements) == 1 and elements[0].isdigit():
                # First line is the entity count
                for line in entity_dict:
                    entity, entity_id = line.strip().split("\t")
                    entity2id[entity] = int(entity_id)
            else:
                entity, entity_id = first_line.strip().split("\t")
                entity2id[entity] = int(entity_id)

                # Now, continue reading and processing the remaining entities
                for line in entity_dict:
                    entity, entity_id = line.strip().split("\t")
                    entity2id[entity] = int(entity_id)

    except FileNotFoundError:
        with open(os.path.join(dataset_folder, "entity2id.tsv"), "r") as entity_dict:
            first_line = entity_dict.readline().strip()
            elements = first_line.split()

            if len(elements) == 1 and elements[0].isdigit():
                # First line is the entity count
                for line in entity_dict:
                    entity_id, entity = line.strip().split("\t")
                    entity2id[entity] = int(entity_id)
            else:
                entity_id, entity = first_line.strip().split("\t")
                entity2id[entity] = int(entity_id)

                # Now, continue reading and processing the remaining entities
                for line in entity_dict:
                    entity_id, entity = line.strip().split("\t")
                    entity2id[entity] = int(entity_id)

    return entity2id


def read_relation2id(dataset_folder):
    relation2id = {}
    try:
        with open(
            os.path.join(dataset_folder, "relation2id.txt"), "r", encoding="utf-8"
        ) as relation_dict:
            first_line = relation_dict.readline().strip()
            elements = first_line.split()

            if len(elements) == 1 and elements[0].isdigit():
                # First line is the relation count
                for line in relation_dict:
                    relation, relation_id = line.strip().split("\t")
                    relation2id[relation] = int(relation_id)
            else:
                relation, relation_id = first_line.strip().split("\t")
                relation2id[relation] = int(relation_id)

                # Now, continue reading and processing the remaining entities
                for line in relation_dict:
                    relation, relation_id = line.strip().split("\t")
                    relation2id[relation] = int(relation_id)

    except:
        with open(
            os.path.join(dataset_folder, "relation2id.tsv"), "r"
        ) as relation_dict:
            first_line = relation_dict.readline().strip()
            elements = first_line.split()

            if len(elements) == 1 and elements[0].isdigit():
                # First line is the relation count
                for line in relation_dict:
                    relation_id, relation = line.strip().split("\t")
                    relation2id[relation] = int(relation_id)
            else:
                relation_id, relation = first_line.strip().split("\t")
                relation2id[relation] = int(relation_id)

                # Now, continue reading and processing the remaining entities
                for line in relation_dict:
                    relation_id, relation = line.strip().split("\t")
                    relation2id[relation] = int(relation_id)

    return relation2id


def read_id2label(dataset_folder, filename):
    id_to_label_mapping = {}
    with open(
        os.path.join(dataset_folder, f"{filename}.txt"), "r", encoding="utf-8"
    ) as text_file:
        for line in text_file:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity_id, entity_label = parts[0], parts[1]
                id_to_label_mapping[entity_id] = entity_label

    return id_to_label_mapping


def save_json(data, filename, indent=4):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyEncoder, ensure_ascii=False, indent=indent)


def load_json(dataset_folder, filename):
    with open(
        os.path.join(dataset_folder, f"{filename}.json"), "r", encoding="utf-8"
    ) as json_file:
        json_data = json.load(json_file)
    return json_data


def replace_entity_ids_with_labels(dataset_folder, text_filename, dict_filename):
    id_to_label_mapping = read_id2label(dataset_folder, text_filename)
    json_data = load_json(dataset_folder, dict_filename)

    def replace_entity_id_with_label(entity_id):
        return id_to_label_mapping.get(entity_id, entity_id)

    def recursive_replace(dictionary):
        modified_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, dict):
                modified_dict[replace_entity_id_with_label(key)] = recursive_replace(
                    value
                )
            else:
                modified_dict[replace_entity_id_with_label(key)] = value
        return modified_dict

    return recursive_replace(json_data)


def parse_wikidata_freebase(dataset_name):
    json_file_path = f"preprocessed_datasets/{dataset_name}/entity2wikidata.json"
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    text_file_path = f"preprocessed_datasets/{dataset_name}/entity2id.txt"
    ent2id = {}
    with open(text_file_path, "r", encoding="utf-8") as text_file:
        for line in text_file:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity_label, entity_id = parts[0], parts[1]
                ent2id[entity_label] = entity_id

    id_to_name_mapping = {}
    for entity_label, entity_id in ent2id.items():
        if entity_label in json_data:
            json_entity = json_data[entity_label]
            if "label" in json_entity:
                label = json_entity["label"]
                id_to_name_mapping[entity_id] = label

    output_file_path = f"preprocessed_datasets/{dataset_name}/id2name.txt"
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for entity_id, name in id_to_name_mapping.items():
            output_file.write(f"{entity_id}\t{name}\n")


def parse_wikidata_codex(dataset_name):
    json_file_path = f"preprocessed_datasets/codex_entities.json"
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    text_file_path = f"preprocessed_datasets/{dataset_name}/entity2id.txt"
    ent2id = {}
    with open(text_file_path, "r", encoding="utf-8") as text_file:
        for line in text_file:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity_label, entity_id = parts[0], parts[1]
                ent2id[entity_label] = entity_id

    id_to_label_mapping = {}
    id_to_wiki_mapping = {}
    for entity_label, entity_id in ent2id.items():
        if entity_label in json_data:
            json_entity = json_data[entity_label]
            if "label" in json_entity:
                label = json_entity["label"]
                id_to_label_mapping[entity_id] = label
            if "wiki" in json_entity:
                wiki = json_entity["wiki"]
                id_to_wiki_mapping[entity_id] = wiki

    output_file_path = f"preprocessed_datasets/{dataset_name}/id2name.txt"
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for entity_id, label in id_to_label_mapping.items():
            output_file.write(f"{entity_id}\t{label}\n")

    output_file_path = f"preprocessed_datasets/{dataset_name}/id2wiki.txt"
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for entity_id, wiki in id_to_wiki_mapping.items():
            output_file.write(f"{entity_id}\t{wiki}\n")


def parse_wikidata_freebase_pykeen(dataset_name):
    json_file_path = f"preprocessed_datasets/{dataset_name}/entity2wikidata.json"
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    text_file_path = f"preprocessed_datasets/{dataset_name}/entity2id.tsv"
    ent2id = {}
    with open(text_file_path, "r", encoding="utf-8") as text_file:
        for line in text_file:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity_id, entity_label = parts[0], parts[1]
                ent2id[entity_label] = entity_id

    id_to_name_mapping = {}
    text_to_name_mapping = {}
    for entity_label, entity_id in ent2id.items():
        if entity_label in json_data:
            json_entity = json_data[entity_label]
            if "label" in json_entity:
                label = json_entity["label"]
                id_to_name_mapping[entity_id] = label
                text_to_name_mapping[entity_label] = label

    output_file_path = f"preprocessed_datasets/{dataset_name}/id2name.txt"
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for entity_id, name in id_to_name_mapping.items():
            output_file.write(f"{entity_id}\t{name}\n")

    output_file_path = f"preprocessed_datasets/{dataset_name}/text2name.txt"
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for entity_label, name in text_to_name_mapping.items():
            output_file.write(f"{entity_label}\t{name}\n")


def parse_wikidata_codex_pykeen(dataset_name):
    json_file_path = f"preprocessed_datasets/codex_entities.json"
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    text_file_path = f"preprocessed_datasets/{dataset_name}/entity2id.tsv"
    ent2id = {}
    with open(text_file_path, "r", encoding="utf-8") as text_file:
        for line in text_file:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity_id, entity_label = parts[0], parts[1]
                ent2id[entity_label] = entity_id

    id_to_label_mapping = {}
    id_to_wiki_mapping = {}
    text_to_label_mapping = {}
    text_to_wiki_mapping = {}
    for entity_label, entity_id in ent2id.items():
        if entity_label in json_data:
            json_entity = json_data[entity_label]
            if "label" in json_entity:
                label = json_entity["label"]
                id_to_label_mapping[entity_id] = label
                text_to_label_mapping[entity_label] = label
            if "wiki" in json_entity:
                wiki = json_entity["wiki"]
                id_to_wiki_mapping[entity_id] = wiki
                text_to_wiki_mapping[entity_label] = wiki

    output_file_path = f"preprocessed_datasets/{dataset_name}/id2name.txt"
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for entity_id, label in id_to_label_mapping.items():
            output_file.write(f"{entity_id}\t{label}\n")

    output_file_path = f"preprocessed_datasets/{dataset_name}/id2wiki.txt"
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for entity_id, wiki in id_to_wiki_mapping.items():
            output_file.write(f"{entity_id}\t{wiki}\n")

    output_file_path = f"preprocessed_datasets/{dataset_name}/text2name.txt"
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for entity_label, label in id_to_label_mapping.items():
            output_file.write(f"{entity_label}\t{label}\n")

    output_file_path = f"preprocessed_datasets/{dataset_name}/text2wiki.txt"
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for entity_label, wiki in id_to_wiki_mapping.items():
            output_file.write(f"{entity_label}\t{wiki}\n")


def determine_kg_format(filename):
    # Read the first line of the file
    with open(filename, "r") as file:
        first_line = file.readline().strip()

        # Check if the first line contains only one unique element (a number)
        elements = first_line.split()
        if len(elements) == 1 and elements[0].isdigit():
            # Remove the first line from the file
            with open(filename, "r") as old_file, open(
                filename + ".tmp", "w"
            ) as new_file:
                next(old_file)  # Skip the first line
                new_file.writelines(old_file)

            # Close the original file
            file.close()
            # Delete the original file
            os.remove(filename)
            # Rename the temp file to the original file
            os.rename(filename + ".tmp", filename)

            # Recall the function to determine the format
            return determine_kg_format(filename)

        # Define patterns to match numeric and textual values
        numeric_pattern = r"^\d+[ \t]+\d+[ \t]+\d+$"
        textual_pattern = r"^[^\d\s\t]+[ \t]+[^\d\s\t]+[ \t]+[^\d\s\t]+$"

    if re.match(numeric_pattern, first_line):
        separator = re.search(r"[ \t]+", first_line).group(0)
        return "ID", separator
    elif re.match(textual_pattern, first_line):
        separator = re.search(r"[ \t]+", first_line).group(0)
        return "Text", separator
    else:
        return "Unknown format", None


def read_as_graph(filename):
    graph = {}
    representation, sep = determine_kg_format(filename)
    with open(filename, "r") as file:
        for line in file:
            elements = line.strip().split(sep)
            # Check if the elements are numeric or textual based on the format
            if representation == "ID":
                entity1, relation, entity2 = map(int, elements)
            elif representation == "Text":
                entity1, relation, entity2 = elements
            else:
                raise ValueError("Unknown format for KG triples.")

            if entity1 not in graph:
                graph[entity1] = {}
            if entity2 not in graph:
                graph[entity2] = {}
            graph[entity1][entity2] = relation
    return graph


def combine_splits(dataset_dir, dataset_name):
    train_file = os.path.join(dataset_dir, dataset_name, "train2id.txt")
    valid_file = os.path.join(dataset_dir, dataset_name, "valid2id.txt")
    test_file = os.path.join(dataset_dir, dataset_name, "test2id.txt")

    with open(train_file, "r") as train:
        train_data = train.read()
    with open(valid_file, "r") as valid:
        valid_data = valid.read()
    with open(test_file, "r") as test:
        test_data = test.read()

    full_graph_data = train_data + valid_data + test_data

    full_graph_file = os.path.join(dataset_dir, dataset_name, "full_graph.txt")
    with open(full_graph_file, "w") as full_graph:
        full_graph.write(full_graph_data)


def parse_pykeen_dict(file_path, format="dict"):
    try:
        # Read the tsv file into a DataFrame
        df = pd.read_csv(file_path, sep="\t")

        if format == "dict":
            # Convert the DataFrame to a Python dictionary with renamed columns
            entity_to_id_dict = df.rename(
                columns={"id": "id_pykeen", "label": "id_preprocessed"}
            )
            entity_to_id_dict = entity_to_id_dict.set_index("id_pykeen")[
                "id_preprocessed"
            ].to_dict()
            return entity_to_id_dict
        elif format == "dataframe":
            # Rename columns in the DataFrame
            df = df.rename(columns={"id": "id_pykeen", "label": "id_preprocessed"})
            return df
        else:
            print("Invalid format specified. Please use 'dict' or 'dataframe'.")
            return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def open_pykeen_model(model_path):
    try:
        trained_model = torch.load(model_path)
        return trained_model
    except FileNotFoundError:
        print(
            f"Trained model file not found. Is it the right relative path?: {model_path}"
        )
        return None
    except Exception as e:
        print(f"An error occurred while loading the trained model: {str(e)}")
        return None


def get_entity_representations(trained_model):
    embeddings = trained_model.entity_representations[0]().detach()
    if embeddings.is_cuda:
        embeddings = embeddings.cpu()
    return embeddings


def generate_adjacency_matrix(num_entities, graph):
    adjacency_matrix = np.zeros((num_entities, num_entities), dtype=np.uint8)

    for entity1 in graph.keys():
        for entity2 in graph.get(entity1, {}):
            adjacency_matrix[
                entity1, entity2
            ] = 1  # If there is a connection, set it to 1

    return adjacency_matrix


def create_shortest_paths_dict(mat, K):
    shortest_paths = {}

    for entity_id in range(mat.shape[0]):
        neighbors = {}
        total_neighbors = 0

        for path_length in range(1, mat.shape[1] + 1):
            # Find the neighbors at the current path length
            path_neighbors = np.where(mat[entity_id] == path_length)[0].tolist()

            # Add the current path neighbors to the total neighbors
            total_neighbors += len(path_neighbors)

            # Check if adding the current path neighbors would exceed the K neighbors threshold
            if total_neighbors >= K:
                neighbors[path_length] = path_neighbors
                break  # Stop if we have enough neighbors

            # Otherwise, add the current path neighbors
            neighbors[path_length] = path_neighbors

        if total_neighbors >= K:
            shortest_paths[entity_id] = neighbors

    return shortest_paths


def unwrap_entity2classes(entities, entity2classes):
    entity2classes_unwrapped = {}
    for entity in entities:
        entity2classes_unwrapped[entity] = entity2classes.get(entity, [])
    return entity2classes_unwrapped


def to_dummy(kg, target_entity):
    dummy_kg = kg.copy()
    mask_h = kg[:, 0] == target_entity
    mask_t = kg[:, 2] == target_entity

    dummy_kg[mask_h, 0] = -1
    dummy_kg[mask_t, 2] = -1

    return dummy_kg


def get_neighbors(kg, entities):
    mask = np.isin(kg[:, 0], entities) | np.isin(kg[:, 2], entities)
    return kg[mask]


def get_all_1_hops(kg, dummy=True):
    # Extract the heads and tails
    heads = kg[:, 0]
    tails = kg[:, 2]

    # Get unique heads and tails
    unique_heads = np.unique(heads)
    unique_tails = np.unique(tails)

    union_heads_tails = np.union1d(unique_heads, unique_tails)

    subgraphs_1_hop = {
        ent: (kg[(kg[:, 0] == ent) | (kg[:, 2] == ent)]) for ent in union_heads_tails
    }

    if dummy:
        subgraphs_1_hop = {
            ent: to_dummy(subgraph, ent) for ent, subgraph in subgraphs_1_hop.items()
        }

    return subgraphs_1_hop


def get_all_2_hops(kg, dummy=True):
    # Extract the heads and tails
    heads = kg[:, 0]
    tails = kg[:, 2]

    # Get unique heads and tails
    unique_heads = np.unique(heads)
    unique_tails = np.unique(tails)

    union_heads_tails = np.union1d(unique_heads, unique_tails)

    subgraphs_1_hop = {
        ent: (kg[(kg[:, 0] == ent) | (kg[:, 2] == ent)]) for ent in union_heads_tails
    }

    subgraphs_2_hops = {}

    for ent in union_heads_tails:
        # Start with the 1-hop neighborhood
        subgraph_1_hop = subgraphs_1_hop[ent]

        # # Collect entities in the 1-hop subgraph
        entities_1_hop = np.union1d(subgraph_1_hop[:, 0], subgraph_1_hop[:, 2])

        # # Get the 1-hop neighbors for all entities in the 1-hop subgraph
        subgraph_2_hop = get_neighbors(kg, entities_1_hop)

        # # Combine the 1-hop and 2-hop subgraphs
        combined_subgraph = np.vstack([subgraph_1_hop, subgraph_2_hop])

        # # Remove duplicates
        combined_subgraph = np.unique(combined_subgraph, axis=0)

        # # Convert to dummy format
        if dummy:
            subgraphs_2_hops[ent] = to_dummy(combined_subgraph, ent)

        else:
            subgraphs_2_hops[ent] = combined_subgraph

    return subgraphs_2_hops


def get_subgraphs_for_entities(kg, entities_of_interest, merge=False, dummy=True):
    # get 1-and-2 hops subgraphs for each entity of interest
    # Create a mapping from entity to its 1-hop triples
    entity_to_triples = defaultdict(set)

    # Create inverse relations mapping
    relations = np.unique(kg[:, 1])
    rel2inv = {rel: rel + len(relations) for rel in relations}

    # Populate entity_to_triples mapping
    for head, rel, tail in kg:
        entity_to_triples[head].add((head, rel, tail))
        entity_to_triples[tail].add((tail, rel2inv[rel], head))

    # Function to generate 2-hop paths
    def generate_2hop_paths(ent, ent_1hop_triples):
        for head, rel, tail in ent_1hop_triples:
            if tail != ent:
                for _, rel2, tail2 in entity_to_triples[tail]:
                    yield (head, rel, rel2, tail2)

    # Function to get subgraphs for a single entity
    def get_subgraphs_for_entity(ent):
        ent_1hop_triples = entity_to_triples[ent]

        # Get 2-hop paths
        ent_2hop_paths = list(generate_2hop_paths(ent, ent_1hop_triples))

        return ent_1hop_triples, ent_2hop_paths

    subgraphs = {}

    # Generate subgraphs for each entity of interest
    # for entity in tqdm(entities_of_interest):
    for entity in entities_of_interest:
        subgraphs[entity] = get_subgraphs_for_entity(entity)

        # Convert sets to arrays
        subgraphs[entity] = (
            np.array(list(subgraphs[entity][0])),
            np.array(list(subgraphs[entity][1])),
        )

        # Replace the head of the first entry of the 3-elements array by -1
        subgraphs[entity][0][:, 0] = -1

        # Remove the first element of the 4-elements array
        subgraphs[entity] = (subgraphs[entity][0], subgraphs[entity][1][:, [1, 2, 3]])

        # Replace the tail of the 3-elements array by -1 in case it is equal to the entity
        subgraphs[entity][1][subgraphs[entity][1][:, 2] == entity, 2] = -1

        if merge:
            # Concatenate the 1-hop and 2-hop subgraphs
            subgraphs[entity] = np.vstack([subgraphs[entity][0], subgraphs[entity][1]])
        else:
            # Only consider the two-hop subgraph
            subgraphs[entity] = subgraphs[entity][1]

    return subgraphs


def get_both_hops_optimized(kg, dummy=True):
    # Create a mapping from entity to its 1-hop triples
    entity_to_triples = defaultdict(set)

    # Create inverse relations mapping
    relations = np.unique(kg[:, 1])
    rel2inv = {rel: rel + len(relations) for rel in relations}

    # Populate entity_to_triples mapping
    for head, rel, tail in kg:
        entity_to_triples[head].add((head, rel, tail))
        entity_to_triples[tail].add((tail, rel2inv[rel], head))

    # Function to generate 2-hop paths
    def generate_2hop_paths(ent, ent_1hop_triples):
        for head, rel, tail in ent_1hop_triples:
            if tail != ent:
                for _, rel2, tail2 in entity_to_triples[tail]:
                    yield (head, rel, rel2, tail2)

    # Function to get subgraphs
    def get_subgraphs(ent):
        ent_1hop_triples = entity_to_triples[ent]

        # Get 2-hop paths
        ent_2hop_paths = list(generate_2hop_paths(ent, ent_1hop_triples))

        return ent_1hop_triples, ent_2hop_paths

    # Get subgraphs for each entity
    subgraphs = {ent: get_subgraphs(ent) for ent in entity_to_triples.keys()}
    subgraphs = dict(sorted(subgraphs.items(), key=lambda x: x[0]))
    # convert sets to arrays
    subgraphs = {
        ent: (np.array(list(subgraphs[ent][0])), np.array(list(subgraphs[ent][1])))
        for ent in subgraphs.keys()
    }
    one_hop_entities = set()  # set of entities with only 1-hop subgraphs
    for ent in subgraphs.keys():
        # replace the head of the first entry of the 3-elements array by -1
        subgraphs[ent][0][:, 0] = -1
        try:
            # remove the first element of the 4-elements array
            subgraphs[ent] = (subgraphs[ent][0], subgraphs[ent][1][:, [1, 2, 3]])
            # replace the tail of the 3-elements array by -1 in case it is equal to ent
            subgraphs[ent][1][subgraphs[ent][1][:, 2] == ent, 2] = -1
        except:
            print("Entity with error:", ent)
            one_hop_entities.add(ent)

    subgraphs = {
        ent: np.vstack([subgraphs[ent][0], subgraphs[ent][1]])
        if ent in subgraphs and ent not in one_hop_entities
        else subgraphs[ent][
            0
        ]  # Handle the case where [ent][0] is used if [ent][1] is missing
        for ent in subgraphs
    }

    return subgraphs


def rbo_score(list1, list2):
    rbo_value = rbo.RankingSimilarity(list1, list2).rbo()
    return rbo_value


def all_pairs_cosine_similarity(embedding_tensor):
    # Normalize each embedding (so they have a magnitude of 1)
    normalized_embeddings = embedding_tensor / embedding_tensor.norm(
        dim=1, keepdim=True
    )

    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

    return similarity_matrix


def get_nodes_with_most_common_neighbors(graph, target_node, k):
    # Calculate common neighbors between the target node and all other nodes
    common_neighbors = {}
    for node in graph.nodes():
        if node != target_node:
            common_neighbors[node] = len(
                list(nx.common_neighbors(graph, target_node, node))
            )

    # Sort nodes by the number of common neighbors in descending order
    sorted_nodes = sorted(common_neighbors.items(), key=lambda x: x[1], reverse=True)

    # Get the top K nodes with the most common neighbors and their counts
    top_k_nodes = {node: count for node, count in sorted_nodes[:k]}

    return top_k_nodes


def get_nodes_with_highest_jaccard_coefficients(graph, target_node, k):
    # Calculate the neighbors of the target node only once
    neighbors_target = set(graph.neighbors(target_node))

    # Calculate Jaccard coefficients between the target node and all other nodes
    jaccard_coefficients = {}
    for node in graph.nodes():
        if node != target_node:
            neighbors_node = set(graph.neighbors(node))
            jaccard_coefficient = len(
                neighbors_target.intersection(neighbors_node)
            ) / len(neighbors_target.union(neighbors_node))
            jaccard_coefficients[node] = jaccard_coefficient

    # Sort nodes by Jaccard coefficient in descending order
    sorted_nodes = sorted(
        jaccard_coefficients.items(), key=lambda x: x[1], reverse=True
    )

    # Get the top K nodes with the highest Jaccard coefficients and their coefficients
    top_k_nodes = {node: coefficient for node, coefficient in sorted_nodes[:k]}

    return top_k_nodes


def jaccard_triples(kg, target_entity):
    dummy_kg = to_dummy(kg, target_entity)

    target_triples = dummy_kg[(dummy_kg[:, 0] == -1) | (dummy_kg[:, 2] == -1)]
    target_to_tails = target_triples[target_triples[:, 0] == -1][:, [1, 2]]
    target_to_tails = {tuple(row) for row in target_to_tails}
    heads_to_target = target_triples[target_triples[:, 2] == -1][:, [0, 1]]
    heads_to_target = {tuple(row) for row in heads_to_target}

    target_neighbors = np.unique(target_triples[:, [0, 2]])
    target_neighbors = target_neighbors[target_neighbors != -1]

    jaccard_coeffs = {}

    for neighbor in target_neighbors:
        dummy_kg_neighbor = to_dummy(kg, neighbor)
        neighbor_triples = dummy_kg_neighbor[
            (dummy_kg_neighbor[:, 0] == -1) | (dummy_kg_neighbor[:, 2] == -1)
        ]
        neighbor_to_tails = neighbor_triples[neighbor_triples[:, 0] == -1][:, [1, 2]]
        neighbor_to_tails = {tuple(row) for row in neighbor_to_tails}
        heads_to_neighbor = neighbor_triples[neighbor_triples[:, 2] == -1][:, [0, 1]]
        heads_to_neighbor = {tuple(row) for row in heads_to_neighbor}

        intersection_to_tails = len(
            [list(row) for row in target_to_tails.intersection(neighbor_to_tails)]
        )
        union_to_tails = len(target_to_tails.union(neighbor_to_tails))

        intersection_heads_to_target = len(
            [list(row) for row in heads_to_target.intersection(heads_to_neighbor)]
        )
        union_heads_to_target = len(heads_to_target.union(heads_to_neighbor))

        jaccard_coeff = (
            (intersection_to_tails + intersection_heads_to_target)
            / (union_to_tails + union_heads_to_target)
            if union_to_tails + union_heads_to_target > 0
            else 0.0
        )

        jaccard_coeffs[neighbor] = round(jaccard_coeff, 3)

    # Sort the dictionary by values in descending order
    sorted_jaccard_coeffs = {
        int(k): v
        for k, v in sorted(
            jaccard_coeffs.items(), key=lambda item: item[1], reverse=True
        )
    }

    return sorted_jaccard_coeffs


def get_k_hop_neighbors(kg, entity, k_hop):
    # Base case: direct triples related to the entity
    neighbors = [entity]
    for k in range(k_hop):
        current_neighbors = np.unique(
            kg[(np.isin(kg[:, 0], neighbors)) | (np.isin(kg[:, 2], neighbors))][
                :, [0, 2]
            ].flatten()
        )
        neighbors.extend(current_neighbors)
    return np.setdiff1d(
        np.unique(neighbors), entity
    )  # Remove duplicates and the original entity


def jaccard_predicates_2hops(kg, target_entity, K_hop_target=2, K_hop_neighbor=2):
    target_neighbors = get_k_hop_neighbors(kg, target_entity, K_hop_target)

    target_triples = kg[
        np.isin(kg[:, 0], target_neighbors) | np.isin(kg[:, 2], target_neighbors)
    ]
    target_to_tails = target_triples[target_triples[:, 0] == target_entity][:, [1, 2]]
    unique_predicates_to_tails = np.unique(target_to_tails[:, 0])

    heads_to_target = target_triples[target_triples[:, 2] == target_entity][:, [0, 1]]
    unique_predicates_to_heads = np.unique(heads_to_target[:, 1])

    jaccard_coeffs = {}

    for neighbor in target_neighbors:
        neighbor_triples = kg[(kg[:, 0] == neighbor) | (kg[:, 2] == neighbor)]

        neighbor_to_tails = neighbor_triples[neighbor_triples[:, 0] == neighbor][
            :, [1, 2]
        ]

        heads_to_neighbor = neighbor_triples[neighbor_triples[:, 2] == neighbor][
            :, [0, 1]
        ]

        unique_predicates_to_tails_neighbor = np.unique(neighbor_to_tails[:, 0])
        unique_predicates_to_heads_neighbor = np.unique(heads_to_neighbor[:, 1])

        intersection_tails = len(
            np.intersect1d(
                unique_predicates_to_tails, unique_predicates_to_tails_neighbor
            )
        )
        union_tails = len(
            np.union1d(unique_predicates_to_tails, unique_predicates_to_tails_neighbor)
        )

        intersection_heads = len(
            np.intersect1d(
                unique_predicates_to_heads, unique_predicates_to_heads_neighbor
            )
        )
        union_heads = len(
            np.union1d(unique_predicates_to_heads, unique_predicates_to_heads_neighbor)
        )

        jaccard_coeff = (
            (intersection_tails + intersection_heads) / (union_tails + union_heads)
            if union_tails + union_heads > 0
            else 0.0
        )

        jaccard_coeffs[neighbor] = round(jaccard_coeff, 3)

    sorted_jaccard_coeffs = {
        int(k): v
        for k, v in sorted(
            jaccard_coeffs.items(), key=lambda item: item[1], reverse=True
        )
    }
    return sorted_jaccard_coeffs


def jaccard_predicates(kg, target_entity):
    target_triples = kg[(kg[:, 0] == target_entity) | (kg[:, 2] == target_entity)]
    target_neighbors = np.unique(target_triples[:, [0, 2]])
    target_neighbors = target_neighbors[target_neighbors != target_entity]

    target_to_tails = target_triples[target_triples[:, 0] == target_entity][:, [1, 2]]
    unique_predicates_to_tails = np.unique(target_to_tails[:, 0])

    heads_to_target = target_triples[target_triples[:, 2] == target_entity][:, [0, 1]]
    unique_predicates_to_heads = np.unique(heads_to_target[:, 1])

    jaccard_coeffs = {}

    for neighbor in target_neighbors:
        neighbor_triples = kg[(kg[:, 0] == neighbor) | (kg[:, 2] == neighbor)]

        neighbor_to_tails = neighbor_triples[neighbor_triples[:, 0] == neighbor][
            :, [1, 2]
        ]

        heads_to_neighbor = neighbor_triples[neighbor_triples[:, 2] == neighbor][
            :, [0, 1]
        ]

        unique_predicates_to_tails_neighbor = np.unique(neighbor_to_tails[:, 0])
        unique_predicates_to_heads_neighbor = np.unique(heads_to_neighbor[:, 1])

        intersection_tails = len(
            np.intersect1d(
                unique_predicates_to_tails, unique_predicates_to_tails_neighbor
            )
        )
        union_tails = len(
            np.union1d(unique_predicates_to_tails, unique_predicates_to_tails_neighbor)
        )

        intersection_heads = len(
            np.intersect1d(
                unique_predicates_to_heads, unique_predicates_to_heads_neighbor
            )
        )
        union_heads = len(
            np.union1d(unique_predicates_to_heads, unique_predicates_to_heads_neighbor)
        )

        jaccard_coeff = (
            (intersection_tails + intersection_heads) / (union_tails + union_heads)
            if union_tails + union_heads > 0
            else 0.0
        )

        jaccard_coeffs[neighbor] = round(jaccard_coeff, 3)

    # Sort the dictionary by values in descending order
    sorted_jaccard_coeffs = {
        int(k): v
        for k, v in sorted(
            jaccard_coeffs.items(), key=lambda item: item[1], reverse=True
        )
    }

    return sorted_jaccard_coeffs
