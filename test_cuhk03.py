#!/usr/bin/env python3
"""
Test script for CUHK03 dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.cuhk03 import CUHK03
from datasets.make_dataloader import make_dataloader
from config import cfg
import argparse

def test_cuhk03():
    """Test the CUHK03 dataset loading"""
    
    # Set up config
    parser = argparse.ArgumentParser(description="Test CUHK03 dataset")
    parser.add_argument("--config_file", default="configs/CUHK03/vit_transreid.yml", help="path to config file", type=str)
    args = parser.parse_args()
    
    # Load config
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    print("Testing CUHK03 dataset...")
    print(f"Dataset root: {cfg.DATASETS.ROOT_DIR}")
    print(f"Dataset name: {cfg.DATASETS.NAMES}")
    
    try:
        # Test dataset initialization (labeled version)
        print("\n=== Testing CUHK03-Labeled ===")
        dataset = CUHK03(root=cfg.DATASETS.ROOT_DIR[0], verbose=True, use_labeled=True)
        
        print(f"\nDataset statistics:")
        print(f"Train: {len(dataset.train)} images, {dataset.num_train_pids} persons")
        print(f"Query: {len(dataset.query)} images, {dataset.num_query_pids} persons")
        print(f"Gallery: {len(dataset.gallery)} images, {dataset.num_gallery_pids} persons")
        print(f"Total cameras: {dataset.num_train_cams}")
        
        # Test dataloader
        print("\nTesting dataloader...")
        train_loader, train_loader_normal, val_loader, num_query, num_classes, cam_num, view_num = make_dataloader(cfg)
        
        print(f"Number of classes: {num_classes}")
        print(f"Number of cameras: {cam_num}")
        print(f"Number of views: {view_num}")
        print(f"Number of query images: {num_query}")
        
        # Test loading a batch
        print("\nTesting batch loading...")
        for i, batch in enumerate(train_loader):
            imgs, pids, camids, viewids = batch
            print(f"Batch {i}: images shape: {imgs.shape}, pids: {pids[:5]}, camids: {camids[:5]}")
            if i >= 2:  # Only test first 3 batches
                break
        
        # Test detected version
        print("\n=== Testing CUHK03-Detected ===")
        dataset_detected = CUHK03(root=cfg.DATASETS.ROOT_DIR[0], verbose=True, use_labeled=False)
        
        print(f"\nDetected dataset statistics:")
        print(f"Train: {len(dataset_detected.train)} images, {dataset_detected.num_train_pids} persons")
        print(f"Query: {len(dataset_detected.query)} images, {dataset_detected.num_query_pids} persons")
        print(f"Gallery: {len(dataset_detected.gallery)} images, {dataset_detected.num_gallery_pids} persons")
                
        print("\n✅ CUHK03 dataset test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cuhk03()
