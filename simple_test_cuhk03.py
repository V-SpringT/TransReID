#!/usr/bin/env python3
"""
Simple test script for CUHK03 dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.cuhk03 import CUHK03

def test_cuhk03_simple():
    """Simple test for CUHK03 dataset loading"""
    
    print("Testing CUHK03 dataset...")
    
    try:
        # Test dataset initialization (labeled version)
        print("\n=== Testing CUHK03-Labeled ===")
        dataset = CUHK03(root='data', verbose=True, use_labeled=True)
        
        print(f"\nDataset statistics:")
        print(f"Train: {len(dataset.train)} images, {dataset.num_train_pids} persons")
        print(f"Query: {len(dataset.query)} images, {dataset.num_query_pids} persons")
        print(f"Gallery: {len(dataset.gallery)} images, {dataset.num_gallery_pids} persons")
        print(f"Total cameras: {dataset.num_train_cams}")
        
        # Test detected version
        print("\n=== Testing CUHK03-Detected ===")
        dataset_detected = CUHK03(root='data', verbose=True, use_labeled=False)
        
        print(f"\nDetected dataset statistics:")
        print(f"Train: {len(dataset_detected.train)} images, {dataset_detected.num_train_pids} persons")
        print(f"Query: {len(dataset_detected.query)} images, {dataset_detected.num_query_pids} persons")
        print(f"Gallery: {len(dataset_detected.gallery)} images, {dataset_detected.num_gallery_pids} persons")
                
        print("\n[SUCCESS] CUHK03 dataset test completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Error testing dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cuhk03_simple()
