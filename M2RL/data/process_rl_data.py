import json
from datasets import Dataset, concatenate_datasets

# Map verbose category names to shorter filenames
category_map = {
    'nano_v3_sft_profiled_dapo17k': 'math',
    'nano_v3_sft_profiled_skywork_no_omni': 'math',
    'nano_v3_sft_profiled_stem_mcqa': 'science',
    'nano_v3_sft_profiled_instruction_following': 'if',
    'nano_v3_sft_profiled_comp_coding_50tests': 'code',
    'nano_v3_sft_profiled_workbench': 'agent',
    'nano_v3_sft_profiled_structured_outputs': 'structured_outputs'
}

# Split dataset by dataset
categories = {}
with open('./raw/Nemotron-3-Nano-RL-Training-Blend/train_complete.jsonl', 'r') as f:
    for line in f:
        item = json.loads(line)
        category = item['dataset']
        if category == 'nano_v3_sft_profiled_structured_outputs':
            continue
        if category not in categories:
            categories[category] = []
        categories[category].append(item)

def process(example, index):
    example['index'] = index
    example['data_source'] = example['dataset']
    message = example['responses_create_params']['input']
    prompt = message[0]['content']
    example['label'] = None

    match example['dataset']:
        case 'nano_v3_sft_profiled_dapo17k' | 'nano_v3_sft_profiled_skywork_no_omni':
            prompt = 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n' + prompt
            example['label'] = example['expected_answer']
            example['metadata'] = {'rm_type': 'deepscaler'}
        case 'nano_v3_sft_profiled_stem_mcqa':
            prompt = 'Answer the following multiple choice question step by step.' + prompt[len('Answer the following multiple choice question.'):]
            example['label'] = example['expected_answer']
            example['metadata'] = {'rm_type': 'gpqa'}
        case 'nano_v3_sft_profiled_instruction_following':
            example['metadata'] = {
                'rm_type': 'ifevalg',
                'prompt_text': example['prompt'],
                'instruction_id_list': example['instruction_id_list'],
                'kwargs': example['kwargs']
            }
        case 'nano_v3_sft_profiled_comp_coding_50tests':
            example['metadata'] = {
                'rm_type': 'unit_test',
                **example['verifier_metadata']
            }
        case 'nano_v3_sft_profiled_workbench':
            example["label"] = None
            # example["tools"] = example['responses_create_params']["tools"]
            example['metadata'] = {
                'rm_type': 'workbench',
                'ground_truth_tool_calls': example["ground_truth"]
            }
    
    message[0]['content'] = prompt
    example['prompt'] = message
    return example

subsets = {}

for category, subset in categories.items():
    data = Dataset.from_list(subset)
    print(category, data)
    data = data.map(
        process,
        remove_columns=[x for x in data.column_names if x not in ['prompt', 'label', 'data_source', 'metadata', 'tools']],
        num_proc=16,
        load_from_cache_file=True,
        with_indices=True
    )
    subsets[category] = data
    if 'dapo17k' in category or 'skywork_no_omni' in category:
        continue
    mapped_name = category_map.get(category, category)
    data.to_parquet(f'./rl_train/{mapped_name}.parquet')

math_data = concatenate_datasets([subsets['nano_v3_sft_profiled_dapo17k'], subsets['nano_v3_sft_profiled_skywork_no_omni']])
math_data = math_data.sort('index')
math_data.to_parquet('./rl_train/math.parquet')
all_data = concatenate_datasets([subset for name, subset in subsets.items() if name not in ['nano_v3_sft_profiled_structured_outputs']])
all_data = all_data.sort('index')
print('Full Training Data:', all_data)
all_data.to_parquet('./rl_train/multi_task.parquet')