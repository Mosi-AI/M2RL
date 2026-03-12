#!/usr/bin/env python3
"""
Merge SFT data - Direct JSONL loading with datasets library concurrency.

This script:
1. Uses datasets library to load JSONL files directly with multiprocessing
2. Processes rows concurrently using datasets.map()
3. Keeps only 'messages' and 'tools' fields
4. Handles schema normalization for consistent output
"""

import os
import tempfile
from pathlib import Path

# Use /dev/shm for temp files (1.6TB RAM disk, supports Unix sockets for multiprocessing)
tempfile.tempdir = '/dev/shm'

# Configure datasets cache to use current directory
cache_dir = Path('cache').resolve()
cache_dir.mkdir(exist_ok=True)

import polars as pl
from typing import Dict, Any, List
import numpy as np
import gc
import orjson
from datasets import load_dataset, Dataset, disable_caching, config, ReadInstruction
from multiprocessing import cpu_count

# Configure datasets to use our cache directory
config.HF_DATASETS_CACHE = str(cache_dir)

# Disable caching for datasets
disable_caching()

# Global: number of processes for parallel operations
NUM_PROC = min(cpu_count(), 64)

# Load HF datasets for competitive coding questions
hf_datasets = {
    "taco": load_dataset("raw/BAAI/TACO", trust_remote_code=True),
    "apps": load_dataset("raw/codeparrot/apps", trust_remote_code=True),
    "code_contests": load_dataset("raw/deepmind/code_contests"),
    "open-r1/codeforces": load_dataset("raw/open-r1/codeforces")
}


def get_question(ds_name: str, split: str, index: int) -> str | None:
    """Get question from HF dataset based on dataset name, split, and index."""
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
    """Replace placeholder '-' in competitive_coding messages with original prompts."""
    messages = row['messages']

    needs_replacement = False
    for msg in messages:
        if msg['role'] == 'user' and msg['content'] == '-':
            needs_replacement = True
            break

    if not needs_replacement:
        return messages

    if 'dataset' not in row or 'split' not in row or 'index' not in row:
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


def clean_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and normalize a single message for consistent schema."""
    clean = {
        'role': msg.get('role', ''),
        'content': msg.get('content', '')
    }

    # Handle non-string content (dict, list, bool, etc.)
    if not isinstance(clean['content'], str):
        if clean['content'] is None:
            clean['content'] = ''
        else:
            clean['content'] = orjson.dumps(clean['content']).decode('utf-8')

    # Only include tool_calls if present and not null
    if 'tool_calls' in msg and msg['tool_calls'] is not None:
        tc = msg['tool_calls']
        if isinstance(tc, str):
            clean['tool_calls'] = tc
        else:
            clean['tool_calls'] = orjson.dumps(tc).decode('utf-8')

    # Only include tool_call_id if present and not null
    if 'tool_call_id' in msg and msg['tool_call_id'] is not None:
        clean['tool_call_id'] = msg['tool_call_id']

    return clean


def process_row(row: Dict[str, Any], is_competitive: bool) -> Dict[str, Any] | None:
    """Process a single row: replace placeholders, validate, and normalize."""
    messages = row.get('messages', [])

    if not isinstance(messages, list) or len(messages) == 0:
        return None

    # Replace competitive coding placeholders if needed
    if is_competitive:
        messages = replace_coding_questions(row)

    # Clean and normalize messages
    cleaned_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        cleaned_messages.append(clean_message(msg))

    if len(cleaned_messages) == 0:
        return None

    # Validate message content
    message_has_content = False
    for msg in cleaned_messages:
        if msg['role'] in ('user', 'assistant'):
            content = msg.get('content', '')
            tool_calls = msg.get('tool_calls', '')
            if len(content) > 0 or len(tool_calls) > 0:
                message_has_content = True
                break

    if not message_has_content:
        return None

    # Build result with normalized tools field
    result = {'messages': cleaned_messages}

    tools = row.get('tools')
    if tools is not None:
        if isinstance(tools, str):
            result['tools'] = tools
        else:
            result['tools'] = orjson.dumps(tools).decode('utf-8')
    else:
        result['tools'] = None

    return result


def process_fn_competitive(row):
    """Process function for competitive coding datasets."""
    result = process_row(row, is_competitive=True)
    if result is None:
        # Consistent schema for filtered rows
        return {'messages': None, 'tools': None, '__filter__': True}
    return {**result, '__filter__': False}


def process_fn_non_competitive(row):
    """Process function for non-competitive coding datasets."""
    result = process_row(row, is_competitive=False)
    if result is None:
        # Consistent schema for filtered rows
        return {'messages': None, 'tools': None, '__filter__': True}
    return {**result, '__filter__': False}


def filter_fn_wrapper(row):
    """Filter function - must be at module level for pickling."""
    return not row['__filter__']


def load_jsonl_with_datasets(jsonl_files: List[Path]) -> Dataset:
    """Load JSONL files using datasets library with concurrent processing."""
    print(f"Loading {len(jsonl_files)} JSONL files with datasets library...")

    # Create a temporary directory with symlinks or just use the first file's directory
    # datasets load_dataset can load from a directory of jsonl files
    temp_data_dir = tempfile.mkdtemp(dir='/dev/shm')

    # Create symlinks to all jsonl files in temp dir
    for i, fpath in enumerate(sorted(jsonl_files)):
        link_path = Path(temp_data_dir) / f"data_{i:05d}.jsonl"
        os.symlink(fpath.resolve(), link_path)

    # Load using datasets - this uses multiple processes internally
    ds = load_dataset(
        "json",
        data_files=[str(p) for p in Path(temp_data_dir).glob("*.jsonl")],
        split="train",
        num_proc=NUM_PROC
    )

    # Cleanup temp dir
    import shutil
    shutil.rmtree(temp_data_dir)

    print(f"✓ Loaded {len(ds):,} rows")
    return ds


def process_dataset_concurrent(ds: Dataset, is_competitive: bool) -> Dataset:
    """Process Dataset rows concurrently."""
    print(f"Processing rows concurrently with {NUM_PROC} processes...")
    print(f"Input dataset columns: {ds.column_names}")

    process_fn = process_fn_competitive if is_competitive else process_fn_non_competitive

    # Process all rows concurrently
    processed_ds = ds.map(
        process_fn,
        num_proc=NUM_PROC,
        desc="Processing rows",
        remove_columns=ds.column_names
    )

    print(f"After map - columns: {processed_ds.column_names}")
    print(f"After map - features: {processed_ds.features}")

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
        print(f"  Downsampling: {original_count:,} → {target_count:,}")
        indices = np.random.choice(original_count, size=target_count, replace=False)
        return ds.select(indices)
    else:
        print(f"  Upsampling: {original_count:,} → {target_count:,}")
        indices = np.random.randint(0, original_count, size=target_count)
        return ds.select(indices)


def main():
    np.random.seed(42)

    raw_dir = Path('raw')

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

    for i, (dir_name, (category, target_count, is_competitive)) in enumerate(dataset_configs.items(), 1):
        print(f"\n[{i}/{len(dataset_configs)}] Processing {dir_name} ({category})...")
        print("-" * 80)

        output_file = f'sft_train/{dir_name}.parquet'
        output_path = Path(output_file)

        if output_path.exists():
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

        jsonl_files = list(dataset_path.glob('*.jsonl'))
        print(f"Found {len(jsonl_files)} files")

        if not jsonl_files:
            continue

        # Step 1: Load JSONL using datasets library directly with concurrency
        ds = load_jsonl_with_datasets(jsonl_files)

        # Step 2: Process rows concurrently
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

        stats[dir_name] = {
            'category': category,
            'original_rows': original_count,
            'target_rows': target_count,
            'sampled_rows': len(sampled_ds),
            'output_file': output_file,
            'status': 'processed'
        }

        temp_files.append(output_file)

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

    print("-" * 80)

    # Merge and shuffle all processed files
    print("\n" + "=" * 80)
    print("Merging and shuffling all processed files...")
    print("=" * 80)

    if temp_files:
        print(f"Reading {len(temp_files)} parquet files...")
        dfs = []
        for pf in temp_files:
            print(f"  Loading {pf}...")
            df = pl.read_parquet(pf)

            # Normalize tools to string for consistent schema
            if 'tools' in df.columns:
                df = df.with_columns([
                    pl.col('tools').cast(pl.String).alias('tools')
                ])

            dfs.append(df)

        # Combine all dataframes
        print("Combining all dataframes...")
        combined_df = pl.concat(dfs, how="diagonal")

        # Shuffle the data
        print("Shuffling data...")
        print(f"Total rows to shuffle: {len(combined_df):,}")

        shuffled_df = combined_df.sample(fraction=1.0, shuffle=True, seed=42)

        # Write to train.parquet
        output_file = 'sft_train/train.parquet'
        print(f"Writing to {output_file}...")
        shuffled_df.write_parquet(output_file, compression='zstd')
        print(f"✓ Saved {len(shuffled_df):,} rows to {output_file}")

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