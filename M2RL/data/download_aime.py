from datasets import load_dataset

data = load_dataset('math-ai/aime24', split='test')

def process(example):
    example['data_source'] = 'aime-2024'
    example['metadata'] = {
        'rm_type': 'deepscaler'
    }
    example['prompt'] = [{
        'role': 'user',
        'content': 'Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} (without quotes) where $Answer is the answer to the problem.\n\n' + example['problem']
    }]
    example['label'] = example['solution'][7:-1]
    return example

data = data.map(process, remove_columns=[x for x in data.column_names if x not in ['prompt', 'label', 'data_source']], num_proc=8)
data.to_parquet('./val/aime-2024.parquet')