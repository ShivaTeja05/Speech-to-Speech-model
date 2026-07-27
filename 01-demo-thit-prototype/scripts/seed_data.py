#!/usr/bin/env python3
"""
Seed Data Script for Apollo Hospital Voice AI
Loads sample_hospital.json into Redis

Usage:
    python scripts/seed_data.py
    
Options:
    --clear     Clear existing data before seeding
    --json-path Path to JSON file (default: data/sample_hospital.json)
"""

import os
import sys
import json
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.redis_store import get_redis, RedisStore


def load_json_data(filepath: str) -> dict:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def clear_existing_data(redis_store: RedisStore):
    """Clear all existing hospital data from Redis"""
    if not redis_store.is_connected:
        print("Redis not connected, cannot clear data")
        return
    
    print("Clearing existing data...")
    
    # Clear config
    redis_store.client.delete("hospital:config")
    
    # Clear doctors
    doctor_ids = redis_store.client.smembers("hospital:doctors:ids")
    for doc_id in doctor_ids:
        redis_store.client.delete(f"hospital:doctors:{doc_id}")
    redis_store.client.delete("hospital:doctors:ids")
    redis_store.client.delete("hospital:doctors:counter")
    
    # Clear departments
    dept_ids = redis_store.client.smembers("hospital:departments:ids")
    for dept_id in dept_ids:
        redis_store.client.delete(f"hospital:departments:{dept_id}")
    redis_store.client.delete("hospital:departments:ids")
    redis_store.client.delete("hospital:departments:counter")
    
    # Clear FAQs
    faq_ids = redis_store.client.smembers("hospital:faqs:ids")
    for faq_id in faq_ids:
        redis_store.client.delete(f"hospital:faqs:{faq_id}")
    redis_store.client.delete("hospital:faqs:ids")
    redis_store.client.delete("hospital:faqs:counter")
    
    print("Existing data cleared")


def seed_config(redis_store: RedisStore, data: dict):
    """Seed hospital configuration"""
    config = data.get('config', {})
    if config:
        redis_store.set_config(config)
        print(f"  Config seeded: {config.get('hospital_name', 'Unknown')}")


def seed_doctors(redis_store: RedisStore, data: dict):
    """Seed doctors"""
    doctors = data.get('doctors', [])
    count = 0
    for doctor in doctors:
        doc_id = redis_store.add_doctor(doctor)
        if doc_id:
            count += 1
    print(f"  Doctors seeded: {count}")


def seed_departments(redis_store: RedisStore, data: dict):
    """Seed departments"""
    departments = data.get('departments', [])
    count = 0
    for dept in departments:
        dept_id = redis_store.add_department(dept)
        if dept_id:
            count += 1
    print(f"  Departments seeded: {count}")


def seed_faqs(redis_store: RedisStore, data: dict):
    """Seed FAQs"""
    faqs = data.get('faqs', [])
    count = 0
    for faq in faqs:
        faq_id = redis_store.add_faq(faq)
        if faq_id:
            count += 1
    print(f"  FAQs seeded: {count}")


def main():
    parser = argparse.ArgumentParser(description='Seed hospital data into Redis')
    parser.add_argument('--clear', action='store_true', help='Clear existing data before seeding')
    parser.add_argument('--json-path', default='data/sample_hospital.json', help='Path to JSON file')
    args = parser.parse_args()
    
    # Resolve path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, args.json_path)
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    
    # Connect to Redis
    print("Connecting to Redis...")
    redis_store = get_redis()
    
    if not redis_store.is_connected:
        print("Error: Could not connect to Redis")
        print("Make sure Redis is running: redis-server")
        sys.exit(1)
    
    print("Connected to Redis!")
    
    # Clear if requested
    if args.clear:
        clear_existing_data(redis_store)
    
    # Load JSON data
    print(f"Loading data from: {json_path}")
    data = load_json_data(json_path)
    
    # Seed data
    print("Seeding data...")
    seed_config(redis_store, data)
    seed_doctors(redis_store, data)
    seed_departments(redis_store, data)
    seed_faqs(redis_store, data)
    
    print("\nDone! Data seeded successfully.")
    
    # Show summary
    print("\nSummary:")
    print(f"  Doctors: {len(redis_store.get_doctors())}")
    print(f"  Departments: {len(redis_store.get_departments())}")
    print(f"  FAQs: {len(redis_store.get_faqs())}")


if __name__ == '__main__':
    main()
