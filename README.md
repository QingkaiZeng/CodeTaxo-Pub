# CodeTaxo-Pub

Code and Data for CodeTaxo

### Data

All datasets used in the Codetaxo paper can be found in the `./data` folder. Under `./data/<datasetName>` (where `<datasetName>` is one of `['wordnet', 'graphine', 'semeval-sci', 'semeval-env', 'semeval-food']`), you will typically find the following files:

- `test_taxonomy_expansion.json`: The original taxonomy expansion dataset.
- `test_definition.json`: A dictionary containing `entity_name: definition` pairs.
- `SimCSE_sampled_test_taxonomy_expansion_50p.json`: The taxonomy expansion dataset after filtering out 50 percent of entities using the Semantic Similarity Filter.
- `SimCSE_sampled_test_taxonomy_expansion_topN.json`: The taxonomy expansion dataset retaining the top-N entities using the Semantic Similarity Filter.


### Model Outputs

The model outputs under different settings can be found in the `./outputs` folder. Each subfolder corresponds to a specific experiment configuration.
