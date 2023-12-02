# Do Similar Entities have Similar Embeddings?

This code repository hosts the datasets, pre-trained embeddings, and scripts to run experiments found in the anonymous submission "Do Similar Entities have Similar Embeddings?" to ESWC 2024.

## Requirements
Required libraries and dependencies can be found in ``requirements.txt``.

## Datasets
The ``preprocessed_datasets`` folder contains already processed datasets with all the required files to effectively carry out experiments.  
In particular, files containing results for the Jaccard coefficients are names ``fast_jaccard2id_train_K=100_hops=X.json`` where X is whether 1 (when considering 1-hop subgraphs) or 0 (when considering 2-hop subgraphs). A mapped version to labels exist, under the same naming convention ``fast_jaccard2labels_train_K=100_hops=X.json``.  
Some of them are not provided (as they exceed the file size limit of GitHub). However, we provide ``fast_jaccard.py`` which uses a fast implementation of Jaccard coefficient computation over subgraphs. This script can be used to generate the required files.

The dataset DBpedia50 is named ``db50_trans`` to highlight the fact that its transductive version is used in the experiments.

## Pre-trained embeddings
Trained models and their respective embeddings can be found under the ``embeddings`` folder. RDF2Vec embeddings were trained using [pyRDF2Vec](https://github.com/IBCNServices/pyRDF2Vec) while other models were trained using [PyKEEN](https://github.com/pykeen/pykeen). For RDF2Vec, we used the default hyperparameters reported in [Portisch et al.](https://www.semantic-web-journal.net/system/files/swj2726.pdf): embeddings of dimension 200, with 2,000 walks maximum, a depth of 4, a window size of 5, and 25 epochs for training word2vec with the continuous skip-gram architecture. For models trained with PyKEEN, we used the provided configuration files, when available. For those datasets with no reported best hyperparameters, we used the hyperparameters reported in the original paper (*e.g.* YAGO4-19K, with best hyperparameters found in [Hubert et al.](https://www.semantic-web-journal.net/system/files/swj3508.pdf)) or performed manual hyperparameter search and kept the sets of hyperparameters leading to the best results on the validation sets (*e.g.* for AIFB).

## Experiments
Running the experiments is quite straightforward: we provide several scripts. Each of them outputs results w.r.t. a separate experimental section in the paper and aims at answering a specific research question (RQ).   
``run_expe_1_2.py`` runs experiments related to RQ1 and RQ2. (Table 2, Figure 3, and Figure 4)  
``run_expe_3.py`` runs experiments related to RQ3. (Table 4)  

## Misc
Additional files are provided to get dataset statistics : ``get_dataset_statistics.py`` and ``get_rel_freq_per_class.py``. These scripts do not need to be run and are not used *per se* in the paper. However, they can provide insights into how predicates are globally distributed across classes, what classes observe a larger set of distinct predicates connecting their respective entities, etc.
