from utils import *
import json
import numpy as np
import os
import argparse
from tqdm import tqdm

DATASET_DIR = "preprocessed_datasets"


def main(args):
    dataset_name = args.dataset_name
    dataset_path = os.path.join(args.path, dataset_name)
    K = args.topk
    hops = args.hops
    split_type = args.split

    # read dataset
    kg = np.loadtxt(os.path.join(dataset_path, f"train2id.txt"), dtype=np.int32)

    if hops == 1:
        subgraphs = get_all_1_hops(kg, dummy=True)
    elif hops == 2:
        subgraphs = get_all_2_hops(kg, dummy=True)
    else:
        if dataset_name in ["codex-m", "FB15K-237", "DB93K", "db50_trans", "db50_ind"]:
            subgraphs = get_both_hops_optimized(kg, dummy=True)
        else:
            subgraphs = get_both_hops_optimized(kg, dummy=True)
            # save the subgraphs in an efficient format
            np.save(
                os.path.join(dataset_path, f"subgraphs_{split_type}_hops={hops}.npy"),
                subgraphs,
            )
    print("Subgraphs extracted")

    # Number of subgraphs
    num_subgraphs = len(subgraphs)

    entity2id = read_entity2id(dataset_path)
    entity_count = len(entity2id)

    if K == -1:
        K = entity_count

    print(K)

    # Jaccard matrix
    jaccard_matrix = np.zeros((entity_count, entity_count))

    # Convert the subgraphs into sets for easier intersection and union operations
    subgraphs = {k: set(map(tuple, v)) for k, v in subgraphs.items()}

    subgraph_sizes = {i: len(subgraphs[i]) for i in subgraphs}
    for i in tqdm(subgraphs):
        for j in (k for k in subgraphs if k > i):  # only compute for j > i
            intersection_size = len(subgraphs[i].intersection(subgraphs[j]))
            union_size = subgraph_sizes[i] + subgraph_sizes[j] - intersection_size
            jaccard_value = 0 if union_size == 0 else intersection_size / union_size
            jaccard_matrix[i][j] = jaccard_matrix[j][i] = jaccard_value

    # Set diagonal to -1
    np.fill_diagonal(jaccard_matrix, -1)

    # For each row, find the indices of the K largest values
    top_k_indices = np.argsort(-jaccard_matrix, axis=1)[:, :K]

    if K == entity_count:
        values_matrix = jaccard_matrix.astype(np.float16)

        values_matrix_quantized = quantize(values_matrix)

        np.save(
            os.path.join(
                dataset_path,
                f"fast_jaccard_quantized_{split_type}_K={K}_hops={hops}.npy",
            ),
            values_matrix_quantized,
        )

        np.savez_compressed(
            os.path.join(
                dataset_path,
                f"fast_jaccard_quantized_{split_type}_K={K}_hops={hops}.npz",
            ),
            values_matrix_quantized,
        )

    else:
        try:
            values_matrix = jaccard_matrix.astype(np.float16)

            values_matrix_quantized = quantize(values_matrix)

            np.save(
                os.path.join(
                    dataset_path,
                    f"fast_jaccard_quantized_{split_type}_K={K}_hops={hops}.npy",
                ),
                values_matrix_quantized,
            )

            np.savez_compressed(
                os.path.join(
                    dataset_path,
                    f"fast_jaccard_quantized_{split_type}_K={K}_hops={hops}.npz",
                ),
                values_matrix_quantized,
            )
        except:
            pass

        jaccard_dict = {
            int(i): {
                int(top_k_indices[i, j]): round(
                    jaccard_matrix[i, top_k_indices[i, j]], 3
                )
                for j in range(K)
            }
            for i in range(jaccard_matrix.shape[0])
        }

        with open(
            os.path.join(
                dataset_path, f"fast_jaccard2id_{split_type}_K={K}_hops={hops}.json"
            ),
            "w",
        ) as json_file:
            json.dump(jaccard_dict, json_file, indent=4)

        try:
            jaccard2labels = replace_entity_ids_with_labels(
                os.path.join(dataset_path),
                "id2name",
                f"fast_jaccard2id_{split_type}_K={K}_hops={hops}",
            )
        except:
            id2entity = {v: k for k, v in entity2id.items()}
            jaccard2labels = {
                id2entity[int(k)]: {id2entity[int(k2)]: v2 for k2, v2 in v.items()}
                for k, v in jaccard_dict.items()
            }

        with open(
            os.path.join(
                dataset_path, f"fast_jaccard2labels_{split_type}_K={K}_hops={hops}.json"
            ),
            "w",
        ) as json_file:
            json.dump(jaccard2labels, json_file, indent=4)


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

    # Parse the arguments
    args = parser.parse_args()

    # Call main function
    main(args)
