import os
import json
import argparse
from tqdm import tqdm
import random

from CodeTaxoUtils import parsing_taxo, load_dataset, get_is_a_relation
from CodeTaxoPromptTemp import construct_NL_prompt, define_prompt_template_NL, define_prompt_template, construct_CodeTaxo_prompt

random.seed(0)

def sample_query_node(taxo):
    entity_list = taxo['entity_list'][:]
    relation_list = taxo['relation_list'][:]
    leaf_nodes = []
    for entity in entity_list:
        parent, child_list = parsing_taxo(taxo=taxo, target=entity)
        if child_list == [] and parent is not None:
            leaf_nodes.append(entity)
    
    try:
        test_node = random.choice(leaf_nodes)
    except:
        print(entity_list)
        print(relation_list)
    parent, _ = parsing_taxo(taxo=taxo, target=test_node)
    # remove the test node from the entity list
    entity_list.remove(test_node)
    # remove the test node and its parent from the relation list
    relation_list.remove([parent, test_node])
            
    return test_node, parent, entity_list, relation_list
    
def genDemoTaxo(args):
    current_dir = os.getcwd()
    if not args.sampling:
        taxonomy_expansion_dataset, entity_definition = load_dataset(args, current_dir=current_dir)
    else:
        if args.percent:
            scale_factor = int(args.scale_factor * 100)
            sampled_taxonomy_expansion_dataset_file_name = f'SimCSE_sampled_test_taxonomy_expansion_{scale_factor}p.json'
        else:
            sampled_taxonomy_expansion_dataset_file_name = f'SimCSE_sampled_test_taxonomy_expansion_top{args.topk}.json'
        taxonomy_expansion_dataset, entity_definition = load_dataset(args, taxo_file_name=sampled_taxonomy_expansion_dataset_file_name, current_dir=current_dir)
    print("Generate Demonstration Taxonomies...")
    
    if args.select_demo:
        sampled_taxonomy_expansion_dataset_file_name = f'SimCSE_sampled_test_taxonomy_expansion_top100.json'
        taxonomy_expansion_dataset, entity_definition = load_dataset(args, taxo_file_name=sampled_taxonomy_expansion_dataset_file_name, current_dir=current_dir)
    
    demoTaxo_list = []
    for each in tqdm(taxonomy_expansion_dataset, desc="Generate Demonstration Taxonomies"):
        demoTaxo_each_taxo = []
        if not args.select_demo:
            for i in range(args.num_demos):
                test_node, test_node_parent, entity_list, relation_list = sample_query_node(each)
                taxo = {}
                taxo['entity_list'] = entity_list
                taxo['test_node'] = test_node
                taxo['root'] = each['root']
                taxo['relation_list'] = relation_list
                taxo['test_node_parent'] = test_node_parent
                taxo['test_node'] = test_node
                demoTaxo_each_taxo.append(taxo)    
            demoTaxo_list.append(demoTaxo_each_taxo)
        else:
            i = 0
            while True:
                try:
                    test_node = each['entity_list'][i]
                except IndexError:
                    # use the last entity in the entity list as the test node for all rest of the demos
                    i = 0
                    continue
                test_node_parent, _ = parsing_taxo(taxo=each, target=test_node)
                if test_node_parent is None:
                    i += 1
                    continue
                entity_list = each['entity_list'][:]
                entity_list.remove(test_node)
                relation_list = each['relation_list'][:]
                relation_list.remove([test_node_parent, test_node])
                taxo = {}
                taxo['entity_list'] = entity_list
                taxo['test_node'] = test_node
                taxo['root'] = each['root']
                taxo['relation_list'] = relation_list
                taxo['test_node_parent'] = test_node_parent
                taxo['test_node'] = test_node
                demoTaxo_each_taxo.append(taxo)
                if len(demoTaxo_each_taxo) == args.num_demos:
                    break
                i += 1
            demoTaxo_list.append(demoTaxo_each_taxo)
            
    return demoTaxo_list
        

def genDemo(args, demoTaxo_list):
    current_dir = os.getcwd()
    if not args.sampling:
        taxonomy_expansion_dataset, entity_definition = load_dataset(args, current_dir=current_dir)
    else:
        if args.percent:
            scale_factor = int(args.scale_factor * 100)
            sampled_taxonomy_expansion_dataset_file_name = f'SimCSE_sampled_test_taxonomy_expansion_{scale_factor}p.json'
        else:
            sampled_taxonomy_expansion_dataset_file_name = f'SimCSE_sampled_test_taxonomy_expansion_top{args.topk}.json'
        taxonomy_expansion_dataset, entity_definition = load_dataset(args, taxo_file_name=sampled_taxonomy_expansion_dataset_file_name, current_dir=current_dir)
    if args.prompt_template_name == 'codetaxo':
        entity_template, prompt_template_head, task_instruction = define_prompt_template(args, current_dir=current_dir)
    elif args.prompt_template_name == 'NL':
        entity_template, task_instruction = define_prompt_template_NL(args, current_dir=current_dir)
    demos_list = []
    
    print("Generate Demonstrations...")
    for each, each_demoTaxo in tqdm(zip(taxonomy_expansion_dataset, demoTaxo_list), desc="Generate Demonstrations"):
        demos_each_taxo = []
        for i in range(args.num_demos):
            demo = {}
            taxo = {}
            # test_node, test_node_parent, entity_list, relation_list = sample_query_node(each)
            taxo['entity_list'] = each_demoTaxo[i]['entity_list']
            taxo['relation_list'] = each_demoTaxo[i]['relation_list']
            taxo['test_node'] = each_demoTaxo[i]['test_node']
            taxo['root'] = each_demoTaxo[i]['root']
            
            if args.prompt_template_name == 'codetaxo':
                prompt_str = construct_CodeTaxo_prompt(args, taxo, entity_template, prompt_template_head, entity_definition)
            elif args.prompt_template_name == 'NL':
                prompt_str = construct_NL_prompt(args, taxo, entity_template, entity_definition)
            
            demo['user_prompt'] = prompt_str
            demo['test_node_parent'] = each_demoTaxo[i]['test_node_parent']
            demo['test_node'] = each_demoTaxo[i]['test_node']
            '''
            if args.gen_explaination:
                demo['explaination'] = get_is_a_relation(demo['test_node'], entity_definition[demo['test_node']], demo['test_node_parent'], entity_definition[demo['test_node_parent']], model=args.model)
            '''
            demos_each_taxo.append(demo)
            
        demos_list.append(demos_each_taxo)
        
    return demos_list

def main_genDemo(args):
    current_dir = os.getcwd()
    if not args.sampling:
        if args.no_definition:
            output_dir = os.path.join(current_dir, 'demos_no_definition')
            if args.select_demo:
                output_dir = os.path.join(current_dir, 'demos_no_definition_SimCSE_select_demo')
        else:
            output_dir = os.path.join(current_dir, 'demos')   
            if args.select_demo:
                output_dir = os.path.join(current_dir, 'demos_SimCSE_select_demo')
    else:
        if args.no_definition:
            output_dir = os.path.join(current_dir, 'demos_no_definition_SimCSE_sampled')
            if args.select_demo:
                output_dir = os.path.join(current_dir, 'demos_no_definition_SimCSE_sampled_select_demo')
        else:
            output_dir = os.path.join(current_dir, 'demos_SimCSE_sampled')
            if args.select_demo:
                output_dir = os.path.join(current_dir, 'demos_SimCSE_sampled_select_demo')

        if args.gen_explaination:
            output_dir = output_dir + '_explaination'
            
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_output_file_name = os.path.join(output_dir, f'{args.dataset}_{args.num_demos}shot_demoTaxo.json')
    if args.sampling:
        if args.percent:
            scale_factor = int(args.scale_factor * 100)
            save_output_file_name = save_output_file_name.replace('demoTaxo.json', f'{scale_factor}p_demoTaxo.json')
        else:
            save_output_file_name = save_output_file_name.replace('demoTaxo.json', f'top{args.topk}_demoTaxo.json')
            
    if args.sample_only:
        demoTaxo_list = genDemoTaxo(args)
        # save the demo taxonomies
        with open(save_output_file_name, 'w') as f:
            json.dump(demoTaxo_list, f)        
    else:
        # check if the demo taxonomies are already generated
        if not os.path.exists(save_output_file_name):
            demoTaxo_list = genDemoTaxo(args)
            # save the demo taxonomies
            with open(save_output_file_name, 'w') as f:
                json.dump(demoTaxo_list, f)
        else:
            with open(save_output_file_name, 'r') as f:
                demoTaxo_list = json.load(f)
        demos_list = genDemo(args, demoTaxo_list)

        # save the demos
        if not args.no_current_taxo:
            save_output_file_name = os.path.join(output_dir, f'{args.dataset}_{args.prompt_template_name}_{args.model}_{args.num_demos}shot_demos.json')
        else:
            save_output_file_name = os.path.join(output_dir, f'{args.dataset}_{args.prompt_template_name}_{args.model}_{args.num_demos}shot_no_current_taxo_demos.json')
            
        if args.sampling:
            if args.percent:
                scale_factor = int(args.scale_factor * 100)
                save_output_file_name = save_output_file_name.replace('demos.json', f'{scale_factor}p_demos.json')
            else:
                save_output_file_name = save_output_file_name.replace('demos.json', f'top{args.topk}_demos.json')
        with open(save_output_file_name, 'w') as f:
            json.dump(demos_list, f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process taxonomy expansion dataset.')
    parser.add_argument('--output_folder', type=str, default='output', help='Output folder name')
    parser.add_argument('--log_folder', type=str, default='log', help='Log folder name')
    parser.add_argument('--data_folder', type=str, default='data', help='Data folder name')
    parser.add_argument('--dataset', type=str, default='wordnet', help='Dataset name, e.g., "wordnet"')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model name, e.g., "gpt-4o"')
    parser.add_argument('--prompt_template_name', type=str, default='codetaxo', help='Prompt template name, e.g., "codetaxo"')
    parser.add_argument('--num_demos', type=int, default=1, help='Number of demos to generate for each test point. Default is 1.')
    parser.add_argument('--no_current_taxo', action='store_true', help='Do not include the current taxonomy in the prompt. Default is False.')
    parser.add_argument('--sample_only', action='store_true', help='Only sample the query node and the demo taxonomy. Default is False.')
    parser.add_argument('--no_definition', action='store_true', help='Do not include the entity definition in the prompt. Default is False.')
    parser.add_argument('--sampling', action='store_true', help='Use SimCSE sampling. Default is False.')
    parser.add_argument('--scale_factor', type=float, default=0.5, help='Scale factor for sampling. Default is 0.5.')
    parser.add_argument('--percent', action='store_true', help='Use scale_factor of the entities in the taxonomy. Default is False.')
    parser.add_argument('--topk', type=int, default=100, help='Number of similar entities to sample. Default is 10.')
    parser.add_argument('--gen_explaination', action='store_true', help='Generate explaination for the taxonomy expansion. Default is False.')
    parser.add_argument('--select_demo', action='store_true', help='Use SimCSE demo selection. Default is False.')
    args = parser.parse_args()
    args.no_current_taxo = True
    print(args)
    
    main_genDemo(args)