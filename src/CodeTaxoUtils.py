import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import openai
import time
import torch
from scipy.spatial.distance import cosine
import networkx as nx
import replicate

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_is_a_relation(entity1, definition1, entity2, definition2, model='gpt-4o', max_retries=5, delay=5):
    prompt = f"""
    Entity 1: {entity1}
    Definition 1: {definition1}
    Entity 2: {entity2}
    Definition 2: {definition2}
    
    Explain why {entity2} is parent concept of {entity1} by one sentence:
    """
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content
        except (openai.OpenAIError, openai.APIConnectionError) as e:
            print(f"Error occurred: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print("Max retries reached. Failed to get response from API.")
                raise

def SimCSE_similarity(node_text_list, entity_list, model, tokenizer, device='cpu'):
    inputs = tokenizer(node_text_list, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu()
    
    similarities = {}
    assert len(entity_list) == len(embeddings) - 1
    for i in range(len(entity_list)):
        similarities[entity_list[i]] = 1 - cosine(embeddings[0], embeddings[i+1])
    return similarities

def SimCSE_topk_similarity(raw_taxo, entity_definition, SimCSE_model, tokenizer, topk=5, device='cpu'):
    test_node = raw_taxo['test_node']
    entity_list = raw_taxo['entity_list']
    
    sampled_taxo = {}
    sampled_taxo['root'] = raw_taxo['root']
    sampled_taxo['test_node'] = test_node
    sampled_taxo['test_node_parent'] = raw_taxo['test_node_parent']

    similarities = {}
    node_text_list = []
    test_node_and_definition = test_node + ': ' + entity_definition[test_node]
    node_text_list.append(test_node_and_definition)
    for entity in entity_list:
        entity_and_definition = entity + ': ' + entity_definition[entity]
        node_text_list.append(entity_and_definition)
        
    similarities = SimCSE_similarity(node_text_list, entity_list, SimCSE_model, tokenizer, device)

    # rank
    similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    # top-k
    similarities = dict(similarities[:topk])
    sampled_taxo['entity_list'] = list(similarities.keys())
    if sampled_taxo['root'] not in sampled_taxo['entity_list']:
        sampled_taxo['entity_list'].append(sampled_taxo['root'])
    sampled_taxo['relation_list'] = []
    for head_entity, tail_entity in raw_taxo['relation_list']:
        if head_entity in sampled_taxo['entity_list'] and tail_entity in sampled_taxo['entity_list']:
            sampled_taxo['relation_list'].append([head_entity, tail_entity])
    
    return sampled_taxo, similarities

def gen_messages(args, prompt, demo_messages=None, messages=None, system_prompt=None):
    if messages is None and system_prompt is None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
    elif messages is None and system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
        ]

    assert type(messages) is list
    if demo_messages is not None:
        if not args.no_current_taxo:
            messages = messages + demo_messages
            messages.append({"role": "user", "content": prompt})
        else:
            if args.prompt_template_name == 'codetaxo':
                current_taxo = prompt.split('\n# creating query node\n\n')[0]
                query = prompt.split('\n# creating query node\n\n')[1]
                query = '\n# creating query node\n\n' + query
            elif args.prompt_template_name == 'NL':
                current_taxo = prompt.split('Query node: ')[0]
                query = prompt.split('Query node: ')[1]
                query = 'Query node: ' + query
            demo_messages[0]['content'] = current_taxo + demo_messages[0]['content']
            messages = messages + demo_messages
            messages.append({"role": "user", "content": query})    
    return messages

def add_message(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages

'''
def call_chat_completion(messages, model='gpt-3.5-turbo'):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
    )
    return response.choices[0].message.content
'''

def convert_openai_messages_to_replicate_input(messages):
    system_prompt = None
    prompt = ""
    
    # Iterate through the messages to construct the system prompt and user-assistant conversation
    for message in messages:
        if message['role'] == 'system':
            system_prompt = message['content']
        elif message['role'] == 'user':
            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message['content']}<|eot_id|>"
        elif message['role'] == 'assistant':
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{message['content']}<|eot_id|>"
    
    # Format the prompt for replicate
    formatted_prompt = f"system\n\n{system_prompt}user\n\n{prompt}assistant\n\n"
    
    return {
        "top_k": 0,
        "top_p": 0.9,
        "prompt": formatted_prompt.strip(),
        "max_tokens": 512,
        "min_tokens": 0,
        "temperature": 0.6,
        "system_prompt": system_prompt,
        "length_penalty": 1,
        "stop_sequences": "<|end_of_text|>,<|eot_id|>",
        "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>{prompt}<|start_header_id|>assistant<|end_header_id|>\n\n",
        "presence_penalty": 1.15,
        "log_performance_metrics": False
    }

def call_chat_completion_replicate(messages, model='meta/meta-llama-3-70b-instruct', max_retries=5, delay=5):
    replicate_input = convert_openai_messages_to_replicate_input(messages)
    for attempt in range(max_retries):
        try:
            output = replicate.run(
                    model,
                    input=replicate_input
                )
            return "".join(output)
        except Exception as e:
            print(f"Error occurred: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print("Max retries reached. Failed to get response from API.")
                raise

def call_chat_completion(messages, model='gpt-3.5-turbo', max_retries=5, delay=5):
    if model in ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']:
        return call_chat_completion_openai(messages, model, max_retries, delay)
    elif model in ['meta-llama-3-70b-instruct', 'codellama-70b-instruct', 'codellama-34b-instruct']:
        model = 'meta/' + 'meta-llama-3-70b-instruct'
        return call_chat_completion_replicate(messages, model, max_retries, delay)
    else:
        raise ValueError(f"Invalid model name: {model}")

def call_chat_completion_openai(messages, model='gpt-3.5-turbo', max_retries=5, delay=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except (openai.OpenAIError, openai.APIConnectionError) as e:
            print(f"Error occurred: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print("Max retries reached. Failed to get response from API.")
                raise

def parsing_taxo(taxo, target):
    root = taxo['root']
    entity_list = taxo['entity_list']
    relation_list = taxo['relation_list']
    
    def find_parent():
        parents = []
        for relation in relation_list:
            if relation[1] == target:
                parents.append(relation[0])
        if len(parents) > 1:
            pass
            # print(f"Error: multiple parents found for {target}")
            # print(f"Parents: {parents}")
        # assert len(parents) <= 1
        return parents[0] if parents else None
    
    def find_child():
        children = []
        for relation in relation_list:
            if relation[0] == target:
                children.append(relation[1])
        return children

    return find_parent(), find_child()

def load_dataset(args, taxo_file_name=None, current_dir=os.getcwd()):
    
    if taxo_file_name is None:
        taxonomy_expansion_dataset_path = os.path.join(current_dir, 'data', args.dataset, 'test_taxonomy_expansion.json')
    else:
        taxonomy_expansion_dataset_path = os.path.join(current_dir, 'data', args.dataset, taxo_file_name)
    with open(taxonomy_expansion_dataset_path) as f:
        taxonomy_expansion_dataset = f.readlines()
        taxonomy_expansion_dataset = [json.loads(taxonomy) for taxonomy in taxonomy_expansion_dataset]
        
    entity_definition_path = os.path.join(current_dir, 'data', args.dataset, 'test_definition.json')
    with open(entity_definition_path) as f:
        entity_definition = json.load(f)
        
    return taxonomy_expansion_dataset, entity_definition

def save_outputs(args, results, prompts, current_dir=os.getcwd()):
    if not args.sampling:
        if args.no_definition:
            output_dir = os.path.join(current_dir, 'output_no_definition')
            log_dir = os.path.join(current_dir, 'log_no_definition')
        else:
            output_dir = os.path.join(current_dir, 'output')
            log_dir = os.path.join(current_dir, 'log')
    else:
        if args.no_definition:
            output_dir = os.path.join(current_dir, 'output_no_definition_SimCSE_sampled')
            log_dir = os.path.join(current_dir, 'log_no_definition_SimCSE_sampled')
        else:
            output_dir = os.path.join(current_dir, 'output_SimCSE_sampled')
            log_dir = os.path.join(current_dir, 'log_SimCSE_sampled')    

    if args.select_demo:
        output_dir = output_dir + '_select_demo'
        log_dir = log_dir + '_select_demo'
    if args.gen_explaination:
        output_dir = output_dir + '_explaination'
        log_dir = log_dir + '_explaination'
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    if not args.no_current_taxo:
        save_output_file_name = os.path.join(output_dir, f'{args.dataset}_{args.prompt_template_name}_{args.model}_{args.num_demos}shot_outputs.json')
    else:
        save_output_file_name = os.path.join(output_dir, f'{args.dataset}_{args.prompt_template_name}_{args.model}_{args.num_demos}shot_no_current_taxo_outputs.json')
    
    if args.sampling:
        if args.percent:
            scale_factor = int(args.scale_factor * 100)
            save_output_file_name = save_output_file_name.replace('outputs.json', f'{scale_factor}p_outputs.json')
        else:
            save_output_file_name = save_output_file_name.replace('outputs.json', f'top{args.topk}_outputs.json')
    
    with open(save_output_file_name, 'w') as f:
        json.dump(results, f)
    print(f"Model outputs saved to {save_output_file_name}")
    
    if not args.no_current_taxo:
        save_log_file_name = os.path.join(log_dir, f'{args.dataset}_{args.prompt_template_name}_{args.model}_{args.num_demos}shot_prompts.json')
    else:
        save_log_file_name = os.path.join(log_dir, f'{args.dataset}_{args.prompt_template_name}_{args.model}_{args.num_demos}shot_no_current_taxo_prompts.json')
        
    if args.sampling:
        if args.percent:
            scale_factor = int(args.scale_factor * 100)
            save_log_file_name = save_log_file_name.replace('prompts.json', f'{scale_factor}p_prompts.json')
        else:
            save_log_file_name = save_log_file_name.replace('prompts.json', f'top{args.topk}_prompts.json')
    with open(save_log_file_name, 'w') as f:
        json.dump(prompts, f)
    print(f"Prompts saved to {save_log_file_name}")
    
class Taxonomy(object):
    def __init__(self, name="", node_list=None, edge_list=None):
        self.name = name
        self.graph = nx.DiGraph()
        self.tx_id2taxon = {}
        self.root = None
        
        if node_list is None:
            node_list = []
        if edge_list is None:
            edge_list = []
        
        for node in node_list:
            self.add_node(node)
        
        for edge in edge_list:
            self.add_edge(edge[0], edge[1])
        
    def __str__(self):
        return f"=== Taxonomy {self.name} ===\nNumber of nodes: {self.graph.number_of_nodes()}\nNumber of edges: {self.graph.number_of_edges()}"
    
    def get_number_of_nodes(self):
        return self.graph.number_of_nodes()

    def get_number_of_edges(self):
        return self.graph.number_of_edges()
    
    def get_nodes(self):
        """
        return: a generator of nodes
        """
        return self.graph.nodes()
    
    def get_edges(self):
        """
        return: a generator of edges
        """
        return self.graph.edges()
    
    def get_root_node(self):
        """
        return: a taxon object
        """
        if not self.root:
            self.root = list(nx.topological_sort(self.graph))[0]
        return self.root
    
    def get_leaf_nodes(self):
        """
        return: a list of taxon objects
        """
        leaf_nodes = []
        for node in self.graph.nodes():
            if self.graph.out_degree(node) == 0:
                leaf_nodes.append(node)
        return leaf_nodes
    
    def get_children(self, parent_node):
        """
        parent_node: a taxon object
        return: a list of taxon object representing the children taxons
        """
        assert parent_node in self.graph, "parent node not in taxonomy"
        return [edge[1] for edge in self.graph.out_edges(parent_node)]
    
    def get_parents(self, child_node):
        """
        child_node: a taxon object
        return: a list of taxon object representing the parent taxons
        """
        assert child_node in self.graph, "child node not in taxonomy"
        return [edge[0] for edge in self.graph.in_edges(child_node)]
    
    def get_siblings(self, child_node):
        """
        child_node: a taxon object
        return: a list of taxon object representing the sibling taxons
        """
        
        assert child_node in self.graph
        parents = self.get_parents(child_node)
        siblings = []
        for parent in parents:
            children = self.get_children(parent)
            for child in children:
                if child != child_node and child not in siblings:
                    siblings.append(child)
        return siblings
        
    def get_descendants(self, parent_node):
        """
        parent_node: a taxon object
        return: a list of taxon object representing the descendant taxons
        """
        assert parent_node in self.graph, "parent node not in taxonomy"
        return list(nx.descendants(self.graph, parent_node))
    
    def get_ancestors(self, child_node):
        """
        child_node: a taxon object
        return: a list of taxon object representing the ancestor taxons
        """
        assert child_node in self.graph, "child node not in taxonomy"
        return list(nx.ancestors(self.graph, child_node))

    def add_node(self, node):
        self.graph.add_node(node)
        
    def add_edge(self, start, end):
        """
        start: a taxon object
        end: a taxon object
        """
        self.graph.add_edge(start, end)

    #These functions are used to calculate Wu&P
    def get_depth(self, node):
        depth = 1
        while node != self.get_root_node():
            parents = self.get_parents(node)
            if not parents:  # In case the node is root or disconnected
                break
            node = parents[0]  # Assuming a single parent as it's a DAG
            depth += 1
        return depth

    def find_LCA(self, node1, node2):
            
        ancestors1 = set(self.get_ancestors(node1))
        ancestors1.add(node1)
        
        ancestors2 = set(self.get_ancestors(node2))
        ancestors2.add(node2)
        
        common_ancestors = ancestors1.intersection(ancestors2)
        # Choose the LCA with the greatest depth
        return max(common_ancestors, key=lambda node: self.get_depth(node), default=None)

    def wu_palmer_similarity(self, pred, ground_truth):
        if pred not in list(self.get_nodes()):
            self.add_node(pred)
            root = self.get_root_node()
            self.add_edge(root, pred)
            depth_lca = 1
            depth_node1 = self.get_depth(pred)
            depth_node2 = self.get_depth(ground_truth)
        else:
            lca = self.find_LCA(pred, ground_truth)
            if lca is None:
                return 0  # No similarity if no LCA

            depth_lca = self.get_depth(lca)
            depth_node1 = self.get_depth(pred)
            depth_node2 = self.get_depth(ground_truth)

        return 2.0 * depth_lca / (depth_node1 + depth_node2)
    
    def is_pred_on_path_to_gt(self, pred, gt):
        paths = list(nx.all_simple_paths(self.graph, source=self.get_root_node(), target=gt))
        
        for path in paths:
            if pred in path:
                return True, path
        
        return False, None
