#!/usr/bin/env python3
"""
Merge SFT data - Maximum speed variant.

This script:
1. Uses Polars for ultra-fast JSONL to parquet conversion
2. Uses datasets library for concurrent row processing with multiprocessing
3. Keeps only 'messages' and 'tools' fields
4. Maximizes parallelism at every stage
"""

import os
import tempfile
from pathlib import Path

# Use /dev/shm for temp files (1.6TB RAM disk, supports Unix sockets for multiprocessing)
# This prevents filling up /tmp (only 119GB) and works with multiprocessing
tempfile.tempdir = '/dev/shm'

# Configure datasets cache to use current directory
cache_dir = Path('cache').resolve()
cache_dir.mkdir(exist_ok=True)

import polars as pl
from typing import Dict, Any, List
import numpy as np
import gc
import orjson
from datasets import load_dataset, Dataset, disable_caching, config, Value, Features
from multiprocessing import cpu_count

# Configure datasets to use our cache directory
config.HF_DATASETS_CACHE = str(cache_dir)

# Disable caching for datasets
disable_caching()

# Global: number of processes for parallel operations
NUM_PROC = min(cpu_count(), 64)  # Cap at 64 to avoid overhead

# Load HF datasets for competitive coding questions
hf_datasets = {
    "taco": load_dataset("raw/BAAI/TACO", trust_remote_code=True),
    "apps": load_dataset("raw/codeparrot/apps", trust_remote_code=True),
    "code_contests": load_dataset("raw/deepmind/code_contests"),
    "open-r1/codeforces": load_dataset("raw/open-r1/codeforces")
}

FEATURES = Features({
    'messages': [
        {
            'content': Value('string'),
            'reasoning_content': Value('string'),
            'role': Value('string'),
            'tool_calls': Value('string')
        }
    ],
    'tools': Value('string'),
    '__filter__': Value('bool')
})


def get_question(ds_name: str, split: str, index: int) -> str | None:
    """
    Get question from HF dataset based on dataset name, split, and index.
    Reference: create_comp_coding.py
    """
    benchmark = hf_datasets[ds_name][split][int(index)]

    if ds_name == "code_contests":
        if not benchmark["description"]:
            return None
        return benchmark["description"]
    elif ds_name in ["taco", "apps"]:
        return benchmark["question"]
    elif ds_name == "open-r1/codeforces":
        if not benchmark["description"]:
            return None
        question = benchmark["description"]
        if benchmark["input_format"]:
            question += "\n\nInput\n\n" + benchmark["input_format"]
        if benchmark["output_format"]:
            question += "\n\nOutput\n\n" + benchmark["output_format"]
        if benchmark["examples"]:
            question += "\n\nExamples"
            for example in benchmark["examples"]:
                if "input" in example:
                    question += "\n\nInput\n\n" + example["input"]
                if "output" in example:
                    question += "\n\nOutput\n\n" + example["output"]
        if benchmark["note"]:
            question += "\n\nNote\n\n" + benchmark["note"]
        return question

    return None


def replace_coding_questions(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Replace placeholder '-' in competitive_coding messages with original prompts.
    Metadata (dataset, split, index) is stored at row level, not message level.
    Only replaces content for messages with role='user'.
    Reference: create_comp_coding.py
    """
    messages = row['messages']

    # Check if there's any message that needs replacement
    needs_replacement = False
    for msg in messages:
        if msg['role'] == 'user' and msg['content'] == '-':
            needs_replacement = True
            break

    if not needs_replacement:
        return messages

    dataset = row['dataset']
    split = row['split']
    index = row['index']

    if dataset not in hf_datasets:
        return messages

    question = get_question(dataset, split, index)
    if not question:
        return messages

    result_messages = []
    for msg in messages:
        if msg['role'] == 'user' and msg['content'] == '-':
            msg = msg.copy()
            msg['content'] = question
        result_messages.append(msg)

    return result_messages


def get_shorted_dict(x):
    """Helper to get a shorted dict for debugging."""
    if isinstance(x, dict):
        return {k: get_shorted_dict(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [get_shorted_dict(v) for v in x]  # Only keep first 2 items in lists
    elif isinstance(x, str):
        return x[:100] + '...' if len(x) > 100 else x  # Shorten long strings
    else:
        return x


def _records_generator(files):
    """Generator that yields records one by one, converting all fields to consistent types."""
    for fpath in files:
        with open(fpath, 'rb') as fp:
            for line in fp:
                r = orjson.loads(line)
                # Validate messages is a list
                msgs = r.get('messages')
                if not isinstance(msgs, list) or len(msgs) == 0:
                    continue

                # Clean messages to remove null tool_calls for consistent typing
                for msg in msgs:
                    if msg.get('tool_calls') is not None:
                        msg['tool_calls'] = orjson.dumps(msg['tool_calls']).decode('utf-8')
                    if msg["role"] == 'tool' and not isinstance(msg["content"], str):
                        msg['content'] = orjson.dumps(msg['content']).decode('utf-8')
                    if "tool_call_id" in msg:
                        msg.pop("tool_call_id")

                # Build a clean record with consistent types
                clean_record = {'messages': msgs}

                # Handle tools field - always convert to JSON string for consistency
                tools = r.get('tools')
                if tools is not None:
                    # Always store as JSON string to avoid type mixing issues
                    clean_record['tools'] = orjson.dumps(tools).decode('utf-8')
                else:
                    clean_record['tools'] = '[]'

                yield clean_record


def jsonl_to_dataset_parallel(jsonl_files: List[Path]) -> Dataset:
    """Streaming JSONL loader with tools as JSON string. Memory efficient."""
    from datasets import Dataset

    total_size = sum(f.stat().st_size for f in jsonl_files) / (1024**3)
    print(f"Loading {len(jsonl_files)} files ({total_size:.2f} GB) with streaming generator...")

    ds = Dataset.from_generator(
        _records_generator,
        gen_kwargs={'files': jsonl_files},
        keep_in_memory=True,
        cache_dir="/dev/shm/"
    )
    print(f"✓ Loaded: {len(ds):,} rows")
    return ds


def jsonl_to_parquet_fast(jsonl_files: List[Path], output_parquet: str):
    """Use Polars for ultra-fast JSONL to parquet conversion."""
    total_size = sum(f.stat().st_size for f in jsonl_files) / (1024**3)  # GB
    print(f"Converting {len(jsonl_files)} files ({total_size:.2f} GB) to parquet with Polars...")

    # Read all JSONL files with schema inference over all rows to avoid NULL type issues
    df = pl.read_ndjson([str(f) for f in jsonl_files], infer_schema_length=None)
    df.write_parquet(output_parquet, compression='zstd')
    print(f"✓ Converted to parquet: {len(df):,} rows")
    del df
    gc.collect()


def process_row(row: Dict[str, Any], is_competitive: bool) -> Dict[str, Any] | None:
    """
    Process a single row: replace placeholders and validate.
    Returns None if row should be filtered out.
    """
    messages = row['messages']

    if len(messages) == 0:
        return None

    # Replace competitive coding placeholders if needed
    if is_competitive:
        messages = replace_coding_questions(row)

    # Validate message content
    for msg in messages:
        if msg["role"] in ("user", "assistant"):
            reasoning_content = msg.get('reasoning_content', '')
            tool_calls = msg.get('tool_calls', [])
            if len(msg['content']) == 0 \
                and (reasoning_content is None or len(reasoning_content) == 0) \
                and (tool_calls is None or len(tool_calls) == 0):
                return None
        if "tool_calls" in msg and msg["tool_calls"] is not None and not isinstance(msg["tool_calls"], str):
            msg["tool_calls"] = orjson.dumps(msg["tool_calls"]).decode('utf-8')
        if "tool_call_id" in msg:
            msg.pop("tool_call_id")
        if 'name' in msg:
            msg.pop('name')

    # Only keep messages and tools
    result = {'messages': messages}
    if 'tools' in row:
        result['tools'] = row['tools']
    if result.get('tools') is None:
        result['tools'] = '[]'
    else:
        result['tools'] = orjson.dumps(result['tools']).decode('utf-8')

    return result


def process_fn_competitive(row):
    """Process function for competitive coding datasets - must be at module level for pickling."""
    result = process_row(row, is_competitive=True)
    if result is None:
        # Must return same keys for consistent schema
        return {'messages': None, 'tools': None, '__filter__': True}
    return {**result, '__filter__': False}


def process_fn_non_competitive(row):
    """Process function for non-competitive coding datasets - must be at module level for pickling."""
    result = process_row(row, is_competitive=False)
    if result is None:
        # Must return same keys for consistent schema
        return {'messages': None, 'tools': None, '__filter__': True}
    return {**result, '__filter__': False}


def filter_fn_wrapper(row):
    """Filter function - must be at module level for pickling."""
    return not row['__filter__']


def process_dataset_concurrent(ds_or_path: Dataset | str, is_competitive: bool) -> Dataset:
    """Process Dataset rows concurrently. Accepts Dataset object or parquet file path."""
    if isinstance(ds_or_path, str):
        print(f"Loading parquet with datasets library...")
        ds = Dataset.from_parquet(ds_or_path)
        print(f"Loaded {len(ds):,} rows")
    else:
        ds = ds_or_path
        print(f"Processing existing dataset: {len(ds):,} rows")

    print(f"Processing rows concurrently with {NUM_PROC} processes...")

    # Choose the appropriate processing function
    process_fn = process_fn_competitive if is_competitive else process_fn_non_competitive

    processed_ds = ds.map(
        process_fn,
        num_proc=NUM_PROC,
        desc="Processing rows",
        remove_columns=ds.column_names,
        features=FEATURES
    )

    # Filter out invalid rows
    print("Filtering invalid rows...")
    filtered_ds = processed_ds.filter(filter_fn_wrapper, num_proc=NUM_PROC)
    filtered_ds = filtered_ds.remove_columns(['__filter__'])

    print(f"After filtering: {len(filtered_ds):,} rows")

    return filtered_ds


def sample_dataset(ds: Dataset, target_count: int) -> Dataset:
    """Sample dataset to target count, repeating if necessary."""
    original_count = len(ds)

    if original_count == 0:
        return ds

    if original_count >= target_count:
        # Downsample: sample without replacement
        print(f"  Downsampling: {original_count:,} → {target_count:,}")
        indices = np.random.choice(original_count, size=target_count, replace=False)
        return ds.select(indices)
    else:
        # Upsample: generate random indices with replacement (some samples will be repeated)
        print(f"  Upsampling: {original_count:,} → {target_count:,}")
        indices = np.random.randint(0, original_count, size=target_count)
        return ds.select(indices)


def main():
    np.random.seed(42)

    raw_dir = Path('raw')

    # Dataset mapping
    dataset_configs = {
        'Nemotron-Math-Proofs-v1': ('math-proofs', 335122, False),
        'Nemotron-Math-v2': ('math', 2950525, False),
        'Nemotron-Science-v1': ('science', 2263340, False),
        'Nemotron-Competitive-Programming-v1': ('code', 3927984, True),
        'Nemotron-Instruction-Following-Chat-v1': ('chat', 4309780, False),
        'Nemotron-Agentic-v1': ('agent', 335122, False),
    }

    temp_files = []
    stats = {}
    skipped_count = 0

    # Process each dataset sequentially
    for i, (dir_name, (category, target_count, is_competitive)) in enumerate(dataset_configs.items(), 1):
        print(f"\n[{i}/{len(dataset_configs)}] Processing {dir_name} ({category})...")
        print("-" * 80)

        # Check if already processed
        output_file = f'sft_train/{dir_name}.parquet'
        output_path = Path(output_file)

        if output_path.exists():
            # Load existing file to get row count
            existing_df = pl.scan_parquet(output_file)
            existing_rows = existing_df.select(pl.len()).collect().item()
            print(f"✓ Already processed: Found {existing_rows} rows in {output_file}")
            print(f"⏭ Skipping processing (use 'rm {output_file}' to reprocess)")

            stats[dir_name] = {
                'category': category,
                'target_rows': target_count,
                'sampled_rows': existing_rows,
                'output_file': output_file,
                'status': 'skipped'
            }

            temp_files.append(output_file)
            skipped_count += 1
            del existing_df
            gc.collect()
            continue

        dataset_path = raw_dir / dir_name / 'data'

        if not dataset_path.exists():
            print(f"Warning: {dataset_path} does not exist, skipping...")
            continue

        # Find all JSONL files
        jsonl_files = list(dataset_path.glob('*.jsonl'))
        print(f"Found {len(jsonl_files)} files")

        if not jsonl_files:
            continue

        # Step 1: Load JSONL - agent data uses parallel loader (handles inconsistent tools schema), others use Polars
        temp_parquet = None
        if category == 'agent':
            print("Using parallel loader for agent data (inconsistent tools schema)...")
            ds = jsonl_to_dataset_parallel(jsonl_files)
        else:
            temp_dir = Path('temp')
            temp_dir.mkdir(exist_ok=True)
            temp_parquet = f'temp/{dir_name}_temp.parquet'
            jsonl_to_parquet_fast(jsonl_files, temp_parquet)
            ds = temp_parquet

        # Step 2: Process rows concurrently with datasets library
        processed_ds = process_dataset_concurrent(ds, is_competitive)
        original_count = len(processed_ds)

        # Step 3: Sample dataset
        print(f"Sampling to {target_count} rows...")
        sampled_ds = sample_dataset(processed_ds, target_count)
        print(f"Sampled rows: {len(sampled_ds):,}")
        del processed_ds
        gc.collect()

        # Step 4: Save to final parquet
        print(f"Saving to {output_file}...")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        sampled_ds.to_parquet(output_file)
        print(f"✓ Saved {len(sampled_ds):,} rows")

        # Cleanup
        if temp_parquet is not None and os.path.exists(temp_parquet):
            os.remove(temp_parquet)

        stats[dir_name] = {
            'category': category,
            'original_rows': original_count,
            'target_rows': target_count,
            'sampled_rows': len(sampled_ds),
            'output_file': output_file,
            'status': 'processed'
        }

        temp_files.append(output_file)

        # Free memory
        del sampled_ds
        gc.collect()

    # Print summary
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"{'Dataset':<45} {'Category':<12} {'Status':<10} {'Rows':<10}")
    print("-" * 80)

    for dir_name in dataset_configs.keys():
        if dir_name in stats:
            stat = stats[dir_name]
            status = stat['status']
            status_display = 'SKIPPED' if status == 'skipped' else 'PROCESSED'
            print(f"{dir_name:<45} {stat['category']:<12} {status_display:<10} {stat['sampled_rows']:<10}")

    print("-" * 80)
    print(f"\nTotal datasets: {len(dataset_configs)}")
    print(f"Processed: {len(dataset_configs) - skipped_count}")
    print(f"Skipped: {skipped_count}")
    if skipped_count > 0:
        print(f"\n💡 Tip: Delete specific parquet files to reprocess individual datasets")

    print("-" * 80)

    # Merge and shuffle all processed files
    print("\n" + "=" * 80)
    print("Merging and shuffling all processed files...")
    print("=" * 80)

    if temp_files:
        print(f"Reading {len(temp_files)} parquet files...")
        # Read all parquet files
        dfs = []
        for pf in temp_files:
            print(f"  Loading {pf}...")
            dfs.append(pl.scan_parquet(pf))

        # Combine all dataframes
        print("Combining all dataframes...")
        combined_df = pl.concat(dfs)

        # Shuffle the data
        print("Shuffling data...")
        total_rows = combined_df.select(pl.len()).collect().item()
        print(f"Total rows to shuffle: {total_rows:,}")

        # Collect and shuffle
        combined_df = combined_df.collect()
        shuffled_df = combined_df.sample(fraction=1.0, shuffle=True, seed=42)

        # Write to train.parquet
        output_file = 'sft_train/train.parquet'
        print(f"Writing to {output_file}...")
        shuffled_df.write_parquet(output_file, compression='zstd')
        print(f"✓ Saved {len(shuffled_df):,} rows to {output_file}")

        # Free memory
        del dfs
        del combined_df
        del shuffled_df
        gc.collect()
    else:
        print("No files to merge.")

    print("\n" + "=" * 80)
    print("✓ Processing complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
