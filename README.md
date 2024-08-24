# CodeTaxo-Pub

Code and Data for CodeTaxo

### Environment setup

```bash
export OPENAI_API_KEY=your_openai_api_key
export REPLICATE_API_TOKEN=your_relicate_api_token
pip install -r requirements.txt
```

### Data

All datasets used in the CodeTaxo paper can be found in the `./data` folder. Under `./data/<datasetName>` (where `<datasetName>` is one of `['wordnet', 'graphine', 'semeval-sci', 'semeval-env', 'semeval-food']`), you will typically find the following files:

- `test_taxonomy_expansion.json`: The original taxonomy expansion dataset.
- `test_definition.json`: A dictionary containing `entity_name: definition` pairs.
- `SimCSE_sampled_test_taxonomy_expansion_50p.json`: The taxonomy expansion dataset after filtering out 50 percent of entities using the Semantic Similarity Filter.
- `SimCSE_sampled_test_taxonomy_expansion_topN.json`: The taxonomy expansion dataset retaining the top-N entities using the Semantic Similarity Filter.

### Model Outputs

The model outputs under different settings can be found in the `./outputs` folder. Each subfolder corresponds to a specific experiment configuration.

### Evaluation

To get the evaluation results based on the existing model outputs:
```bash
python src/eval.py --dataset $dataset --model $model --prompt_template_name $prompt_template_name --num_demos $num_demos --sampling --select_demo
```

- `--dataset`: Dataset name, `[wordnet, graphine, semeval-sci, semeval-env, semeval-food]` (default: wordnet)
- `--model`: Model name, e.g., `[gpt-4o, gpt-4o-mini, gpt-3.5-turbo, meta-llama-3-70b-instruct, codellama-34b-instruct, codellama-70b-instruct]`  (default: gpt-4o)
- `--prompt_template_name`: Use CodeTaxo prompt or Natural Language prompt, `[codetaxo, NL]` (default: codetaxo)
- `--num_demos`: Number of demos (default: 1)
- `--no_definition`: Exclude the entity definition in the prompt (default: False)
- `--sampling`: Use SimCSE sampling (default: False)
- `--scale_factor`: Scale factor for SimCSE sampling, keep the top `scale_factor * 100` % entities, used when `--percent` is set (default: 0.5)
- `--percent`: Keep scale_factor of the entities in the taxonomy (default: False)
- `--topk`: Number of similar entities to sample, used when `--percent` is not set(default: 100)
- `--gen_explaination`: Generate explanation for the taxonomy expansion (default: False)
- `--select_demo`: Use demo selection (default: False)

### Run

```bash
python src/main.py --dataset $dataset --model $model --prompt_template_name $prompt_template_name --num_demos $num_demos --sampling --select_demo
```

- `--dataset`: Dataset name, `[wordnet, graphine, semeval-sci, semeval-env, semeval-food]` (default: wordnet)
- `--model`: Model name, e.g., `[gpt-4o, gpt-4o-mini, gpt-3.5-turbo, meta-llama-3-70b-instruct, codellama-34b-instruct, codellama-70b-instruct]`  (default: gpt-4o)
- `--prompt_template_name`: Use CodeTaxo prompt or Natural Language prompt, `[codetaxo, NL]` (default: codetaxo)
- `--num_demos`: Number of demos (default: 1)
- `--no_definition`: Exclude the entity definition in the prompt (default: False)
- `--sampling`: Use SimCSE sampling (default: False)
- `--scale_factor`: Scale factor for SimCSE sampling, keep the top `scale_factor * 100` % entities, used when `--percent` is set (default: 0.5)
- `--percent`: Keep scale_factor of the entities in the taxonomy (default: False)
- `--topk`: Number of similar entities to sample, used when `--percent` is not set(default: 100)
- `--gen_explaination`: Generate explanation for the taxonomy expansion (default: False)
- `--select_demo`: Use demo selection (default: False)