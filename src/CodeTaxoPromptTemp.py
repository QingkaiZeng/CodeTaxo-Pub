from CodeTaxoUtils import parsing_taxo
import os

def define_prompt_template_NL(args, current_dir):
    entity_template = "{entity}: {definition}; parent: {parent}; children: {child_list}."
    task_instruction = "Given the current taxonomy, find the parent of the query node. Please note that the query node may be a new node not in the current taxonomy. The parent of given query node always exists, so do not generate 'none' or 'not found'. You only need to answer the entity name and do not generate any additional content or comments."
    
    return entity_template, task_instruction

def define_prompt_template_HF(args, current_dir):
    pass

def define_prompt_template_TP(args, current_dir):
    pass

def define_prompt_template(args, current_dir=os.getcwd()):
    
    entity_template = "{entity_vname} = Entity(name='{entity}', description='{description}', parent={parent}, child={child_list})"
    prompt_template_path = os.path.join(current_dir, 'src', f'{args.prompt_template_name}.txt')
    prompt_template_head = open(prompt_template_path).read()
    if not args.gen_explaination:
        task_instruction = "Complete the next line of code according to the comments and the given code snippet. You need to find the parent of the query node in the given current taxonomy and use the add_parent function. The parent of given query node always exists in the given current taxonomy, so do NOT generate node that is NOT in the given current taxonomy. Note that you only need to complete the next ONE line of code, do not generate any additional content or comments."
    else:
        task_instruction = "Complete the next line of code according to the comments and the given code snippet. You need to find the parent of the query node in the given current taxonomy and use the add_parent function. The parent of given query node always exists in the given current taxonomy, so do NOT generate node that is NOT in the given current taxonomy. Note that you only need to complete the next ONE line of code, do not generate any additional content or comments."
        
    '''
        task_instruction = "Complete the next line of code according to the comments and the given code snippet. You need to find the parent of the query node in the given current taxonomy and use the add_parent function. The parent of given query node always exists in the given current taxonomy, so do NOT generate node that is NOT in the given current taxonomy. Note that you only need to complete the next ONE line of code an one-line explanation to explain why it is the parent node of the given query node, DO NOT generate any additional content."
    '''
    
    return entity_template, prompt_template_head, task_instruction

def construct_NL_prompt(args, taxo, entity_template, entity_definition, main_loop=False):
    entity_list = taxo['entity_list']
    test_node = taxo['test_node']
    prompt_template = 'Current taxonomy:\n'
    
    # construct the current taxonomy
    if not args.no_current_taxo or main_loop:
        for entity in entity_list:
            parent, child_list = parsing_taxo(taxo=taxo, target=entity)
            if args.no_definition:
                description = ''
            else:
                description = entity_definition[entity]
            entity_instance = entity_template.replace('{entity}', entity).replace('{definition}', description).replace('{parent}', str(parent)).replace('{child_list}', str(child_list))
            prompt_template += entity_instance + '\n'
    
    # construct the query node
        prompt_template += f"\nQuery node: {test_node}\n"
    else:
        prompt_template = f"\nQuery node: {test_node}\n"
    
    # Find the parent of query node
    prompt_template += "The parent of query node: "
    return prompt_template

def construct_CodeTaxo_prompt(args, taxo, entity_template, prompt_template_head, entity_definition, main_loop=False):
    entity_list = taxo['entity_list']
    prompt_template = prompt_template_head
    test_node = taxo['test_node']
    
    # construct the current taxonomy
    if not args.no_current_taxo or main_loop:
        for entity in entity_list:
            parent, child_list = parsing_taxo(taxo=taxo, target=entity)
            if args.no_definition:
                description = ''
            else:
                description = entity_definition[entity]
            entity_instance = entity_template.replace('{entity_vname}', entity.replace(' ', '_')).replace('{entity}', entity).replace('{description}', description).replace('{parent}', str(parent)).replace('{child_list}', str(child_list))
            prompt_template += entity_instance + '\n'

    # construct the query node
        prompt_template += "\n# creating query node\n\n"
    else:
        prompt_template = "\n# creating query node\n\n"
        
    parent = None
    child_list = []
    if args.no_definition:
        description = ''
    else:
        description = entity_definition[test_node]
    entity_instance = entity_template.replace('{entity_vname}', test_node.replace(' ', '_')).replace('{entity}', test_node).replace('{description}', description).replace('{parent}', str(parent)).replace('{child_list}', str(child_list))
    prompt_template += entity_instance + '\n'
    
    # Find the parent of query node
    if not main_loop:
        prompt_template += "\n# Finding the parent of query node"
    else:
        if args.gen_explaination:
            prompt_template += "\n# Finding the parent of query node in the first line, and then generate a comment to explain why it is the parent of the given query node in the next line\n"
        else:
            prompt_template += "\n# Finding the parent of query node\n"
    return prompt_template