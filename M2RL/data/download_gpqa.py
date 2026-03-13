import random
from datasets import load_dataset

data = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', split='train')

def process(x):
    gold_index = random.randint(0, 3)
    choices = [x["Incorrect Answer 1"], x["Incorrect Answer 2"], x["Incorrect Answer 3"]]
    choices.insert(gold_index, x["Correct Answer"])
    instruction = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering."
    query_template = "{Instruction}\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    query = query_template.format(
        A=choices[0].strip(),
        B=choices[1].strip(),
        C=choices[2].strip(),
        D=choices[3].strip(),
        Question=x["Question"].strip(),
        Instruction=instruction,
    )

    x['data_source'] = 'gpqa'
    x['prompt'] = query
    x['label'] = 'ABCD'[gold_index]
    x['metadata'] = {
        'rm_type': 'gpqa',
        'valid_letters': 'ABCD',
        'correct_letter': x['label']
    }
    return x

data = data.map(process, remove_columns=[x for x in data.column_names if x not in ['prompt', 'label', 'data_source']], num_proc=8)
data.to_parquet('./val/gpqa.parquet')