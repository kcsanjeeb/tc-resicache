#!/usr/bin/env python3
"""
RedSync Results Analyzer
Parses JSONL experiment results and generates summary reports.

Usage:
    python analyze_results.py results/
    python analyze_results.py results/cifar10_redsync.jsonl
"""

import json
import sys
import glob
from pathlib import Path
from collections import defaultdict


def parse_jsonl(filepath):
    """Parse a JSONL file and return list of records."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def analyze_cifar(records, method_name):
    """Analyze CIFAR experiment results."""
    if not records:
        return None
    
    # Get final epoch results
    final_record = records[-1]
    
    # Calculate average epoch time (stable phase: last 20 epochs)
    stable_records = records[-20:] if len(records) >= 20 else records
    avg_epoch_time = sum(r.get('train_time_sec', 0) for r in stable_records) / len(stable_records)
    
    # Get accuracy
    test_acc = final_record.get('test_acc', final_record.get('accuracy', 0)) * 100
    
    # Get cache hit rate if available
    cache_hit_rate = final_record.get('cache_hit_rate', 0) * 100
    
    # Get memory usage
    memory_mb = final_record.get('memory_mb', final_record.get('peak_memory_mb', 0))
    memory_gb = memory_mb / 1024 if memory_mb else 0
    
    return {
        'method': method_name,
        'final_accuracy': test_acc,
        'avg_epoch_time_sec': avg_epoch_time,
        'cache_hit_rate': cache_hit_rate,
        'memory_gb': memory_gb,
        'total_epochs': len(records)
    }


def analyze_imagenet(records, method_name):
    """Analyze ImageNet experiment results."""
    if not records:
        return None
    
    final_record = records[-1]
    
    stable_records = records[-10:] if len(records) >= 10 else records
    avg_epoch_time = sum(r.get('train_time_sec', 0) for r in stable_records) / len(stable_records)
    
    top1 = final_record.get('test_top1', final_record.get('top1', 0)) * 100
    top5 = final_record.get('test_top5', final_record.get('top5', 0)) * 100
    cache_hit_rate = final_record.get('cache_hit_rate', 0) * 100
    
    return {
        'method': method_name,
        'top1_accuracy': top1,
        'top5_accuracy': top5,
        'avg_epoch_time_sec': avg_epoch_time,
        'cache_hit_rate': cache_hit_rate,
        'total_epochs': len(records)
    }


def analyze_glue(records, method_name):
    """Analyze GLUE experiment results."""
    if not records:
        return None
    
    final_record = records[-1]
    
    total_time = sum(r.get('train_time_sec', 0) for r in records)
    
    accuracy = final_record.get('accuracy', 0) * 100
    f1 = final_record.get('f1', 0) * 100
    cache_hit_rate = final_record.get('cache_hit_rate', 0) * 100
    
    return {
        'method': method_name,
        'accuracy': accuracy,
        'f1': f1,
        'total_time_sec': total_time,
        'cache_hit_rate': cache_hit_rate,
        'total_epochs': len(records)
    }


def generate_report(results_dir):
    """Generate comprehensive report from all results."""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return
    
    # Collect all JSONL files
    jsonl_files = list(results_dir.glob('*.jsonl'))
    
    if not jsonl_files:
        print(f"No JSONL files found in {results_dir}")
        return
    
    print("=" * 60)
    print("RedSync Experimental Results Summary")
    print("=" * 60)
    print()
    
    # Categorize results
    cifar_results = defaultdict(dict)
    imagenet_results = defaultdict(dict)
    glue_results = defaultdict(dict)
    
    for filepath in jsonl_files:
        filename = filepath.stem
        records = parse_jsonl(filepath)
        
        if 'cifar10' in filename or 'cifar100' in filename:
            dataset = 'cifar10' if 'cifar10' in filename else 'cifar100'
            method = filename.replace(f'{dataset}_', '')
            cifar_results[dataset][method] = analyze_cifar(records, method)
            
        elif 'imagenet' in filename:
            dataset = 'imagenet100' if 'imagenet100' in filename else 'imagenet1k'
            method = filename.replace(f'{dataset}_', '')
            imagenet_results[dataset][method] = analyze_imagenet(records, method)
            
        elif 'glue' in filename or 'mrpc' in filename or 'sst2' in filename or 'mnli' in filename:
            task = filename.split('_')[1] if '_' in filename else 'unknown'
            method = filename.split('_')[-1] if '_' in filename else 'unknown'
            glue_results[task][method] = analyze_glue(records, method)
    
    # Print CIFAR results
    if cifar_results:
        print("CIFAR Results:")
        print("-" * 60)
        for dataset, methods in sorted(cifar_results.items()):
            print(f"\n{dataset.upper()}:")
            print(f"{'Method':<20} {'Accuracy':<12} {'Time/Epoch':<15} {'Speedup':<10} {'Hit Rate':<10}")
            print("-" * 70)
            
            # Get baseline time for speedup calculation
            baseline_time = None
            if 'nocache' in methods:
                baseline_time = methods['nocache']['avg_epoch_time_sec']
            
            for method, data in sorted(methods.items()):
                if data:
                    speedup = baseline_time / data['avg_epoch_time_sec'] if baseline_time else 1.0
                    print(f"{method:<20} {data['final_accuracy']:>10.2f}%  "
                          f"{data['avg_epoch_time_sec']:>12.1f}s  "
                          f"{speedup:>8.2f}x  "
                          f"{data['cache_hit_rate']:>8.1f}%")
        print()
    
    # Print ImageNet results
    if imagenet_results:
        print("ImageNet Results:")
        print("-" * 60)
        for dataset, methods in sorted(imagenet_results.items()):
            print(f"\n{dataset.upper()}:")
            print(f"{'Method':<20} {'Top-1':<10} {'Top-5':<10} {'Time/Epoch':<15} {'Hit Rate':<10}")
            print("-" * 70)
            for method, data in sorted(methods.items()):
                if data:
                    print(f"{method:<20} {data['top1_accuracy']:>8.2f}%  "
                          f"{data['top5_accuracy']:>8.2f}%  "
                          f"{data['avg_epoch_time_sec']:>12.1f}s  "
                          f"{data['cache_hit_rate']:>8.1f}%")
        print()
    
    # Print GLUE results
    if glue_results:
        print("GLUE Results:")
        print("-" * 60)
        for task, methods in sorted(glue_results.items()):
            print(f"\n{task.upper()}:")
            print(f"{'Method':<20} {'Accuracy':<12} {'F1':<10} {'Total Time':<15} {'Hit Rate':<10}")
            print("-" * 70)
            for method, data in sorted(methods.items()):
                if data:
                    print(f"{method:<20} {data['accuracy']:>10.2f}%  "
                          f"{data['f1']:>8.2f}%  "
                          f"{data['total_time_sec']:>12.0f}s  "
                          f"{data['cache_hit_rate']:>8.1f}%")
        print()
    
    # Print summary for paper
    print("=" * 60)
    print("Summary for Paper (Copy-Paste Friendly)")
    print("=" * 60)
    print()
    
    if cifar_results:
        for dataset, methods in sorted(cifar_results.items()):
            if 'nocache' in methods and 'redsync' in methods:
                baseline = methods['nocache']
                redsync = methods['redsync']
                speedup = baseline['avg_epoch_time_sec'] / redsync['avg_epoch_time_sec']
                acc_diff = redsync['final_accuracy'] - baseline['final_accuracy']
                
                print(f"{dataset.upper()}:")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Accuracy: {redsync['final_accuracy']:.2f}% (baseline: {baseline['final_accuracy']:.2f}%, diff: {acc_diff:+.2f}%)")
                print(f"  Cache Hit Rate: {redsync['cache_hit_rate']:.1f}%")
                print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_directory>")
        print("       python analyze_results.py <jsonl_file>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_file():
        # Analyze single file
        records = parse_jsonl(path)
        print(f"File: {path}")
        print(f"Records: {len(records)}")
        if records:
            print(f"First record: {json.dumps(records[0], indent=2)}")
            print(f"Last record: {json.dumps(records[-1], indent=2)}")
    else:
        # Generate full report
        generate_report(path)
