#!/usr/bin/env python3
"""
Batch Preprocessing for PRISM Dataset
Generate elevation_map PKL files for all sequences (Parallel Processing)
"""

import os
import sys
import yaml
import pickle
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tools.gt_processor import GTProcessor


def process_single_pcd(args):
    """
    Process a single PCD file (for parallel processing)

    Args:
        args: tuple of (config_path, pcd_file_path, output_file_path, filter_by_histogram)

    Returns:
        tuple: (success: bool, timestamp: str, error_msg: str or None)
    """
    config_path, pcd_file, output_file, filter_by_histogram = args

    try:
        # Initialize processor in each worker process
        processor = GTProcessor(config_path)
        
        # Process PCD
        if processor.include_normal_gt:
            ele_gt, ele_mask, normal_gt, normal_mask, _ = processor.process(pcd_file, dx=0, filter_by_histogram=filter_by_histogram)
            if not isinstance(normal_gt, torch.Tensor):
                normal_gt = torch.tensor(normal_gt, dtype=torch.float32)
            if not isinstance(normal_mask, torch.Tensor):
                normal_mask = torch.tensor(normal_mask, dtype=torch.bool)

            normal_gt = normal_gt.to(torch.float32)
            normal_mask = normal_mask.to(torch.bool)
            normal_gt[~normal_mask] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)  # Set invalid normals to point upwards

        else:
            ele_gt, ele_mask, _ = processor.process(pcd_file, dx=0, filter_by_histogram=filter_by_histogram)
        
        # Ensure data types match RSRD format
        if not isinstance(ele_gt, torch.Tensor):
            ele_gt = torch.tensor(ele_gt, dtype=torch.float32)
        if not isinstance(ele_mask, torch.Tensor):
            ele_mask = torch.tensor(ele_mask, dtype=torch.bool)

        # Convert to correct dtype if needed
        ele_gt = ele_gt.to(torch.float32)
        ele_mask = ele_mask.to(torch.bool)

        # Set invalid pixels to 0.0 (RSRD convention)
        ele_gt[~ele_mask] = 0.0

        # Save as PKL (RSRD format)
        with open(output_file, 'wb') as f:
            if processor.include_normal_gt:
                pickle.dump([ele_gt, ele_mask, normal_gt, normal_mask], f)
            else:
                pickle.dump([ele_gt, ele_mask], f)
        
        return (True, Path(pcd_file).stem, None)

    except Exception as e:
        return (False, Path(pcd_file).stem, str(e))


def process_sequence(processor, sequence_dir, output_dir, config_path, num_workers=8, filter_by_histogram=True):
    """
    Process all PCD files in a sequence (parallel)

    Args:
        processor: GTProcessor instance (for info only, not used in parallel)
        sequence_dir: Path to sequence directory
        output_dir: Path to elevation_map output directory
        config_path: Path to config.yaml
        num_workers: Number of parallel workers
        filter_by_histogram: Enable histogram-based filtering (default: True)

    Returns:
        num_success: Number of successfully processed files
        num_failed: Number of failed files
    """
    sequence_path = Path(sequence_dir)
    pcd_dir = sequence_path / "lidar_accum_scan"

    if not pcd_dir.exists():
        print(f"[ERROR] PCD directory not found: {pcd_dir}")
        return 0, 0

    # Get all PCD files
    pcd_files = sorted(pcd_dir.glob("*.pcd"))

    if not pcd_files:
        print(f"[WARNING] No PCD files found in {pcd_dir}")
        return 0, 0

    # Quiet mode - only show progress bar

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for parallel processing
    tasks = []
    for pcd_file in pcd_files:
        timestamp = pcd_file.stem
        output_file = output_path / f"{timestamp}.pkl"
        tasks.append((str(config_path), str(pcd_file), str(output_file), filter_by_histogram))

    # Process in parallel
    num_success = 0
    num_failed = 0
    failed_files = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_pcd, task): task for task in tasks}

        # Process results with progress bar
        for future in tqdm(as_completed(futures), total=len(tasks),
                          desc=f"{sequence_path.name}", ncols=100):
            success, timestamp, error_msg = future.result()

            if success:
                num_success += 1
            else:
                num_failed += 1
                failed_files.append((f"{timestamp}.pcd", error_msg))
                tqdm.write(f"[ERROR] Failed {timestamp}.pcd: {error_msg}")

    # Print summary
    status = "✓" if num_failed == 0 else "⚠"
    print(f"{status} {sequence_path.name}: {num_success}/{len(pcd_files)} success")

    if failed_files:
        for fname, error in failed_files[:3]:  # Show first 3 errors only
            print(f"  ✗ {fname}: {error}")
        if len(failed_files) > 3:
            print(f"  ... and {len(failed_files) - 3} more errors")

    return num_success, num_failed


def main():
    """Main function"""
    # Load config
    # Load config using absolute path
    script_dir = Path(__file__).parent.resolve()
    config_path = script_dir.parent / "config/gt_config.yaml"
    
    if not config_path.exists():
        print(f"[ERROR] Cannot find config at: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        print(f"[ERROR] Config at {config_path} loaded as None. Check file encoding.")
        return
    
    # Get dataset bases (multiple datasets)
    dataset_bases = config['input'].get('dataset_bases', [])
    
    # Fallback to old format if dataset_bases not found
    if not dataset_bases:
        if 'dataset_base' in config['input']:
            dataset_bases = [config['input']['dataset_base']]
        else:
            print("[ERROR] No dataset_bases specified in config.yaml")
            return
    
    # Get optional sequence filter
    sequence_filter = config['input'].get('sequences', None)
    
    # Determine number of workers (use 80% of CPU cores)
    num_workers = max(1, int(mp.cpu_count() * 0.8))

    # Filtering options
    filter_by_histogram = True  # Enable histogram-based filtering

    print(f"\n{'='*80}")
    print("PRISM Dataset Batch Preprocessing (Parallel)")
    print(f"{'='*80}")
    print(f"Config: {config_path}")
    print(f"CPU cores: {mp.cpu_count()}")
    print(f"Workers: {num_workers}")
    print(f"Datasets: {len(dataset_bases)}")
    print(f"Histogram filtering: {filter_by_histogram}")
    if sequence_filter:
        print(f"Sequence filter: {sequence_filter}")
    else:
        print(f"Sequence filter: None (process all sequence_*)")
    
    # Resolve dataset_bases against PRISM_DATASET_ROOT if set
    dataset_root = Path(os.environ['PRISM_DATASET_ROOT']) if 'PRISM_DATASET_ROOT' in os.environ else None

    # Collect all sequences from all datasets
    all_sequences = []
    for dataset_base_str in dataset_bases:
        p = Path(dataset_base_str)
        dataset_base = p if p.is_absolute() else (dataset_root / p if dataset_root else p)
        if not dataset_base.exists():
            print(f"\n[WARNING] Dataset not found: {dataset_base}")
            continue
        
        print(f"\n  Dataset: {dataset_base}")
        
        # Auto-detect all sequences
        sequences = sorted([d for d in dataset_base.iterdir() 
                          if d.is_dir() and d.name.startswith('sequence_')])
        
        # Apply sequence filter if specified
        if sequence_filter:
            sequences = [s for s in sequences if s.name in sequence_filter]
            print(f"  Sequences (filtered): {len(sequences)}")
        else:
            print(f"  Sequences: {len(sequences)}")
        
        for seq_path in sequences:
            pcd_dir = seq_path / "lidar_accum_scan"
            if pcd_dir.exists():
                num_files = len(list(pcd_dir.glob("*.pcd")))
                print(f"    - {seq_path.name}: {num_files} files")
                all_sequences.append((dataset_base, seq_path))
            else:
                print(f"    - {seq_path.name}: [NO PCD FILES]")
    
    print(f"{'='*80}")
    print(f"Total sequences to process: {len(all_sequences)}")
    print(f"{'='*80}")
    
    if not all_sequences:
        print("\n[ERROR] No sequences found to process!")
        return
    
    # Initialize processor (for info only)
    print("\nInitializing GTProcessor...")
    processor = GTProcessor(config_path)
    
    # Process each sequence
    total_success = 0
    total_failed = 0
    
    for dataset_base, sequence_dir in all_sequences:
        output_dir = sequence_dir / processor.gt_folder_name
        num_success, num_failed = process_sequence(
            processor, sequence_dir, output_dir, config_path, num_workers,
            filter_by_histogram=filter_by_histogram
        )

        total_success += num_success
        total_failed += num_failed
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total datasets: {len(dataset_bases)}")
    print(f"Total sequences processed: {len(all_sequences)}")
    print(f"Total files processed: {total_success + total_failed}")
    print(f"  Success: {total_success}")
    print(f"  Failed:  {total_failed}")
    if total_success + total_failed > 0:
        print(f"  Success rate: {total_success/(total_success+total_failed)*100:.2f}%")
    print(f"{'='*80}\n")
    
    # Verify output
    print("Verifying generated PKL files...")
    for dataset_base_str in dataset_bases:
        dataset_base = Path(dataset_base_str)
        if not dataset_base.exists():
            continue
        
        print(f"\n  Dataset: {dataset_base.name}")
        sequences = sorted([d for d in dataset_base.iterdir() 
                          if d.is_dir() and d.name.startswith('sequence_')])
        
        for seq_path in sequences:
            pkl_dir = seq_path / processor.gt_folder_name
            if pkl_dir.exists():
                pkl_files = list(pkl_dir.glob("*.pkl"))
                print(f"    {seq_path.name}: {len(pkl_files)} PKL files")
                
                # Check first file format
                if pkl_files:
                    with open(pkl_files[0], 'rb') as f:
                        data = pickle.load(f)
                    ele_gt, ele_mask = data
                    print(f"      Sample: {pkl_files[0].name}")
                    print(f"        ele_gt shape: {ele_gt.shape}, dtype: {ele_gt.dtype}")
                    print(f"        ele_mask shape: {ele_mask.shape}, dtype: {ele_mask.dtype}")
    
    print(f"\n{'='*80}")
    print("Batch preprocessing complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    # Ensure proper multiprocessing on Linux
    mp.set_start_method('spawn', force=True)
    main()