import os
import json
import argparse
from tqdm import tqdm

from CodeTaxoUtils import gen_messages, add_message, call_chat_completion, parsing_taxo, load_dataset, save_outputs
from nl import main_NL
from genDemo import main_genDemo
from CodeTaxoPromptTemp import construct_CodeTaxo_prompt, define_prompt_template
from eval import eval

def demo_to_messages(demo, task_instruction):
    user_prompt = demo['user_prompt']
    test_node_parent = demo['test_node_parent']
    test_node = demo['test_node']
    user = {"role": "user", "content": task_instruction + '\n\n' + user_prompt}
    '''
    if args.gen_explaination:
        explaination = demo['explaination']
        explaination = f'# Explaination: {explaination}'
        assistant = {"role": "assistant", "content": f'{explaination}\n{test_node}.add_parent({test_node_parent})'}
    else:
        assistant = {"role": "assistant", "content": f'{test_node}.add_parent({test_node_parent})'}
    '''
    assistant = {"role": "assistant", "content": f'{test_node}.add_parent({test_node_parent})'}
    return [user, assistant]
    
def main(args):
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
    entity_template, prompt_template_head, task_instruction = define_prompt_template(args, current_dir=current_dir)
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
    
    if args.num_demos >= 1:
        if not args.no_current_taxo:
            file_path = os.path.join(output_dir, f'{args.dataset}_{args.prompt_template_name}_{args.model}_{args.num_demos}shot_demos.json')
        else:
            file_path = os.path.join(output_dir, f'{args.dataset}_{args.prompt_template_name}_{args.model}_{args.num_demos}shot_no_current_taxo_demos.json')
            
        if args.sampling:
            if args.percent:
                scale_factor = int(args.scale_factor * 100)
                file_path = file_path.replace('demos.json', f'{scale_factor}p_demos.json')
            else:
                file_path = file_path.replace('demos.json', f'top{args.topk}_demos.json')
        with open(file_path, 'r') as f:
            demos_list = json.load(f)
    
    results = []
    prompts = []

    print("Processing dataset...")
    for i, each in tqdm(enumerate(taxonomy_expansion_dataset), desc="Processing taxonomy data"):
        
        few_shot_demos = demos_list[i]
        few_shot_demo_messages = []
        for shot in range(args.num_demos):
            demo = few_shot_demos[shot]
            few_shot_demo_messages += demo_to_messages(demo, task_instruction)

        prompt_str = construct_CodeTaxo_prompt(args, each, entity_template, prompt_template_head, entity_definition, main_loop=True)
        prompt = gen_messages(args, prompt=task_instruction + '\n\n' + prompt_str, demo_messages=few_shot_demo_messages, system_prompt=task_instruction)
        prompts.append(prompt)
        res = call_chat_completion(messages=prompt, model=args.model)
        results.append(res)
        

    save_outputs(args, results, prompts, current_dir=current_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process taxonomy expansion dataset.')
    parser.add_argument('--output_folder', type=str, default='output', help='Output folder name')
    parser.add_argument('--log_folder', type=str, default='log', help='Log folder name')
    parser.add_argument('--data_folder', type=str, default='data', help='Data folder name')
    parser.add_argument('--dataset', type=str, default='wordnet', help='Dataset name, e.g., "wordnet"')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model name, e.g., "gpt-4o"')
    parser.add_argument('--prompt_template_name', type=str, default='codetaxo', help='Prompt template name, e.g., "codetaxo", "NL"')
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
    print('-----------------------Demo Generation Started--------------------------')
    print(args)
    main_genDemo(args)
    print('-----------------------Demo Generation Finished--------------------------')
    print('\n\n')
    print('-----------------------Taxonomy Expansion Started--------------------------')
    
    if args.prompt_template_name == 'NL':
        print('NL')
        main_NL(args)
    else:
        main(args)
        
    print('-----------------------Taxonomy Expansion Finished-------------------------')
    print('\n\n')
    print('-----------------------Evaluation Started--------------------------')
    eval(args)
    print('-----------------------Evaluation Finished--------------------------')

        
    