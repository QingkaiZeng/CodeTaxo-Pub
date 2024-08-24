import argparse
import json
import os
import re
from CodeTaxoUtils import Taxonomy

def print_prompt(prompt):
    system_prompt = prompt[0]['content']
    user_prompt = prompt[1]['content']
    print(f"System: {system_prompt}")
    print(f"User: {user_prompt}")

def extract_code(input_text):
    pattern = re.compile(r'\b[\w\s-]+\.[\w\s-]+\([^\)]*\)')
    code_lines = pattern.findall(input_text)
    extracted_code = '\n'.join(code_lines)
    return extracted_code

def print_output_ground_truth(output, extract_output, test_node, test_node_parent):
    print("Output:")
    print(output)
    print('Extracted Output:')
    print(extract_output)
    print("\nGround Truth:")
    print('Test Node: ', test_node)
    print('Test Node Parent: ', test_node_parent)
    
def count_error_types(outputs, data, args):
    errors = {"None/none/Not found/No data": 0, "Parent entity not in entity list": 0, "Parent entity in list but incorrect": 0, 'test_node_parent not in entity_list': 0 ,'correct': 0}

    for i in range(len(outputs)):
        output = outputs[i]
        taxo = data[i]
        entity_list = taxo['entity_list']
        relation_list = taxo['relation_list']
        test_node_parent = taxo['test_node_parent']

        if args.prompt_template_name == 'codetaxo':
            # extract_output = extract_code(output)
            if args.gen_explaination:
                split_output = output.split('#')
                try:
                    extract_output = [line for line in split_output if 'add_parent' in line][0].strip('\n')
                except:
                    extract_output = ''
            else:
                extract_output = output
            if extract_output == '':
                continue
            try:
                child = extract_output.split('.add_parent(')[0].replace('_', ' ')
                parent = extract_output.split('.add_parent(')[1].split(')')[0].replace('_', ' ')
            except:
                continue

        elif args.prompt_template_name == 'NL':
            parent = output.lower()

        if parent.lower() in ["none", "not found", "no data"]:
            errors["None/none/Not found/No data"] += 1
        elif parent.lower() not in [e.lower() for e in entity_list] and parent.lower().replace(' ', '_') not in [e.lower() for e in entity_list]:
            # print('output: ', parent)
            # print('ground truth: ', taxo['test_node'], '--', test_node_parent)
            # print(parent.lower() == test_node_parent.lower())
            errors["Parent entity not in entity list"] += 1
        elif parent.lower() != test_node_parent.lower() or parent.lower().replace(' ', '_') != test_node_parent.lower().replace(' ', '_'):
            errors["Parent entity in list but incorrect"] += 1
        elif parent.lower() == test_node_parent.lower() or parent.lower().replace(' ', '_') == test_node_parent.lower().replace(' ', '_'):
            errors['correct'] += 1
        else:
            pass
            '''
            print('raw output: ', output)
            print('extracted output: ', extract_output)
            print("Output: ", child, '--',parent)
            print("GT: ", taxo['test_node'], '--',test_node_parent)
            '''
        if test_node_parent.lower() not in [e.lower() for e in entity_list]:
            errors['test_node_parent not in entity_list'] += 1
            

    return errors
    
def eval(args):
    if not args.no_current_taxo:
        output_file_name = f'{args.dataset}_{args.prompt_template_name}_{args.model}_{args.num_demos}shot_outputs.json'
        prompt_file_name = f'{args.dataset}_{args.prompt_template_name}_{args.model}_{args.num_demos}shot_prompts.json'
    else:
        output_file_name = f'{args.dataset}_{args.prompt_template_name}_{args.model}_{args.num_demos}shot_no_current_taxo_outputs.json'
        prompt_file_name = f'{args.dataset}_{args.prompt_template_name}_{args.model}_{args.num_demos}shot_no_current_taxo_prompts.json'
        
    if args.sampling:
        if args.percent:
            scale_factor = int(args.scale_factor * 100)
            output_file_name = output_file_name.replace('outputs.json', f'{scale_factor}p_outputs.json')
            prompt_file_name = prompt_file_name.replace('prompts.json', f'{scale_factor}p_prompts.json')
        else:
            output_file_name = output_file_name.replace('outputs.json', f'top{args.topk}_outputs.json')
            prompt_file_name = prompt_file_name.replace('prompts.json', f'top{args.topk}_prompts.json')

    current_dir = '.'
    if not args.sampling:
        dataset_file_name = 'test_taxonomy_expansion.json'
        if args.no_definition:
            args.output_folder = 'output_no_definition'
            args.log_folder = 'log_no_definition'
    else:
        if args.percent:
            scale_factor = int(args.scale_factor * 100)
            dataset_file_name = f'SimCSE_sampled_test_taxonomy_expansion_{scale_factor}p.json'
        else:
            dataset_file_name = f'SimCSE_sampled_test_taxonomy_expansion_top{args.topk}.json'
        # dataset_file_name = 'SimCSE_sampled_test_taxonomy_expansion.json'
        if args.no_definition:
            args.output_folder = 'output_no_definition_SimCSE_sampled'
            args.log_folder = 'log_no_definition_SimCSE_sampled'
        else:
            args.output_folder = 'output_SimCSE_sampled'
            args.log_folder = 'log_SimCSE_sampled'
            
    if args.select_demo:
        args.output_folder = args.output_folder + '_select_demo'
        args.log_folder = args.log_folder + '_select_demo'
        
    if args.gen_explaination:
        args.output_folder = args.output_folder + '_explaination'
        args.log_folder = args.log_folder + '_explaination'

    outputs = json.load(open(os.path.join(current_dir, args.output_folder, output_file_name)))
    prompts = json.load(open(os.path.join(current_dir, args.log_folder, prompt_file_name)))
    with open(os.path.join(current_dir, args.data_folder, args.dataset, dataset_file_name)) as f:
        data = f.readlines()
        data = [json.loads(taxonomy) for taxonomy in data]
        
    with open(os.path.join(current_dir, args.data_folder, args.dataset, 'test_taxonomy_expansion.json')) as f:
        data_not_sampled = f.readlines()
        data_not_sampled = [json.loads(taxonomy) for taxonomy in data_not_sampled]

    results = []
    num_invalid_format = 0
    num_using_add_child = 0
    false_cases = []
    wu_p_results = []
    for i in range(len(outputs)):
        taxo = data[i]
        taxo_not_sampled = data_not_sampled[i]
        output = outputs[i]
        prompt = prompts[i]

        test_node = taxo['test_node']
        test_node_parent = taxo['test_node_parent']
        
        entity_list = taxo_not_sampled['entity_list']
        entity_list = [entity.lower() for entity in entity_list]
        relation_list = taxo_not_sampled['relation_list']
        relation_list = [(relation[0].lower(), relation[1].lower()) for relation in relation_list]

        taxonomy = Taxonomy(node_list=entity_list, edge_list=relation_list)
        assert test_node_parent.lower() in entity_list, f"Parent entity {test_node_parent} not in entity list"

        if args.prompt_template_name == 'codetaxo':
            if args.gen_explaination:
                split_output = output.split('#')
                try:
                    extract_output = [line for line in split_output if 'add_parent' in line][0].strip('\n')
                except:
                    extract_output = ''
            else:
                extract_output = output
            # extract_output = extract_code(output)
            if extract_output == '':
                results.append(0)
                wu_p_results.append(taxonomy.wu_palmer_similarity('Invalid', test_node_parent))
                num_invalid_format += 1
                continue
            
            try:
                child = extract_output.split('.add_parent(')[0].replace('_', ' ')
                parent = extract_output.split('.add_parent(')[1].split(')')[0].replace('_', ' ')
            except:
                results.append(0)
                wu_p_results.append(taxonomy.wu_palmer_similarity('Invalid', test_node_parent))
                num_using_add_child += 1
                continue
            
            if parent.lower() == test_node_parent.lower():
                results.append(1)
            elif parent.lower().replace(' ', '_') == test_node_parent.lower().replace(' ', '_'):
                parent = parent.replace(' ', '_')
                test_node_parent = test_node_parent.replace(' ', '_')
                results.append(1)
            else:
                results.append(0)
                false_cases.append([prompt, output, test_node, test_node_parent, taxo])

            wu_p_results.append(taxonomy.wu_palmer_similarity(parent.lower(), test_node_parent.lower()))
            
        elif args.prompt_template_name == 'NL':
            if output.lower() == test_node_parent.lower():
                results.append(1)
            else:
                results.append(0)

            wu_p_results.append(taxonomy.wu_palmer_similarity(output.lower(), test_node_parent.lower()))
    
    acc = (sum(results) / len(results)) * 100
    wu_p = sum(wu_p_results) / len(wu_p_results) * 100
    '''
    if args.prompt_template_name == 'codetaxo':
        print(f'len(results): {len(results)}')
        print(f'sum(results): {sum(results)}')
        print(f"Number of invalid format: {num_invalid_format}")
        print(f"Number of using add_child: {num_using_add_child}")
    '''
    print(f"Accuracy: {acc:.2f}")
    print(f"Wu-Palmer Similarity: {wu_p:.2f}")
    
    '''
    if args.prompt_template_name == 'codetaxo':
        errors = count_error_types(outputs, data, args)
        print("Error Types:")
        no_in_list = errors["Parent entity not in entity list"]
        in_list_incorrect = errors["Parent entity in list but incorrect"]
        for error_type, count in errors.items():
            print(f"{error_type}: {count}")
    '''
    
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
    print('---------------------------------------------------------')
    args.no_current_taxo = True
    print(args)
    print('---------------------------------------------------------')
    eval(args)
    print('\n\n')


