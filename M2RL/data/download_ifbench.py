from datasets import load_dataset, Features, Value

data = load_dataset('allenai/IFBench_test', split='train')

def process(example):
    example['label'] = None
    example['metadata'] = {
        'rm_type': 'ifbench',
        'prompt_text': example['prompt'],
        'instruction_id_list': example['instruction_id_list'],
        'kwargs': example['kwargs']
    }
    return example

old_features = data.features
data = data.map(
    process,
    remove_columns=[x for x in data.column_names if x not in ['prompt', 'label', 'data_source']],
    num_proc=8,
    features=Features({
        'prompt': old_features['prompt'],
        'label': Value('null'),
        'metadata': {
            'prompt_text': Value('string'),
            'rm_type': Value('string'),
            'instruction_id_list': old_features['instruction_id_list'],
            'kwargs': old_features['kwargs']
        }
    })
)
data.to_parquet('./val/IFBench_test.parquet')