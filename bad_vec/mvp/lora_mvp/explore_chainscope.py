#!/usr/bin/env python3
"""
Explore ChainScope dataset to find unfaithful CoT examples,
particularly focusing on GSM8K problems.
"""

import json
import requests
import yaml
import os
from pathlib import Path

print("="*80)
print("EXPLORING CHAINSCOPE DATASET")
print("="*80)

# ChainScope repo structure based on the paper
base_url = "https://raw.githubusercontent.com/jettjaniak/chainscope/main/"

# Paths mentioned in the repo
paths_to_check = [
    "chainscope/data/questions/",  # Question datasets (yamls starting with "wm")
    "chainscope/data/cot_responses/",  # CoT responses
    "chainscope/data/problems/",  # Restoration errors datasets
    "chainscope/data/cot_paths/",  # CoT paths
    "chainscope/data/properties/",  # Properties for generating questions
]

def fetch_github_dir(url):
    """Fetch directory listing from GitHub API"""
    # Convert raw URL to API URL
    api_url = url.replace("https://raw.githubusercontent.com/", "https://api.github.com/repos/")
    api_url = api_url.replace("/main/", "/contents/")
    
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# First, try to get the repository structure
print("\nChecking repository structure...")
repo_api_url = "https://api.github.com/repos/jettjaniak/chainscope/contents/chainscope/data"

try:
    response = requests.get(repo_api_url)
    if response.status_code == 200:
        data_dirs = response.json()
        print(f"Found {len(data_dirs)} directories in chainscope/data/:")
        for item in data_dirs:
            if item['type'] == 'dir':
                print(f"  - {item['name']}/")
except Exception as e:
    print(f"Error fetching repo structure: {e}")

# Check for question datasets
print("\n" + "="*60)
print("CHECKING FOR QUESTION DATASETS")
print("="*60)

questions_url = "https://api.github.com/repos/jettjaniak/chainscope/contents/chainscope/data/questions"
try:
    response = requests.get(questions_url)
    if response.status_code == 200:
        files = response.json()
        yaml_files = [f for f in files if f['name'].endswith('.yaml') or f['name'].endswith('.yml')]
        print(f"Found {len(yaml_files)} YAML files in questions/:")
        
        # Check for GSM-related files
        gsm_files = []
        for f in yaml_files[:10]:  # Check first 10
            print(f"  - {f['name']}")
            if 'gsm' in f['name'].lower():
                gsm_files.append(f)
        
        if gsm_files:
            print(f"\nFound {len(gsm_files)} GSM-related files!")
            
            # Download and check one
            for gsm_file in gsm_files[:1]:
                print(f"\nDownloading {gsm_file['name']}...")
                file_response = requests.get(gsm_file['download_url'])
                if file_response.status_code == 200:
                    data = yaml.safe_load(file_response.text)
                    if isinstance(data, dict):
                        print(f"  Keys: {list(data.keys())[:5]}")
                        # Show sample
                        for key, value in list(data.items())[:2]:
                            print(f"  Example {key}: {str(value)[:200]}...")
                    elif isinstance(data, list):
                        print(f"  Contains {len(data)} entries")
                        if data:
                            print(f"  First entry: {data[0]}")
        else:
            print("\nNo GSM-specific files found. Checking file contents...")
            
            # Download a sample file to check content
            if yaml_files:
                sample_file = yaml_files[0]
                print(f"\nChecking {sample_file['name']} for GSM content...")
                file_response = requests.get(sample_file['download_url'])
                if file_response.status_code == 200:
                    content = file_response.text[:1000]
                    if 'gsm' in content.lower() or 'math' in content.lower():
                        print("  Found math/GSM references!")
                    print(f"  Content preview: {content[:300]}...")
except Exception as e:
    print(f"Error: {e}")

# Check for CoT responses
print("\n" + "="*60)
print("CHECKING FOR COT RESPONSES")
print("="*60)

cot_url = "https://api.github.com/repos/jettjaniak/chainscope/contents/chainscope/data/cot_responses"
try:
    response = requests.get(cot_url)
    if response.status_code == 200:
        files = response.json()
        print(f"Found {len(files)} files in cot_responses/:")
        
        json_files = [f for f in files if f['name'].endswith('.json')]
        for f in json_files[:5]:
            print(f"  - {f['name']}")
        
        # Check a sample for unfaithful CoT
        if json_files:
            sample = json_files[0]
            print(f"\nChecking {sample['name']} for unfaithful CoT examples...")
            file_response = requests.get(sample['download_url'])
            if file_response.status_code == 200:
                data = json.loads(file_response.text)
                print(f"  Structure: {type(data)}")
                if isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())[:10]}")
                    # Check for GSM problems
                    gsm_count = sum(1 for k in data.keys() if 'gsm' in str(k).lower())
                    print(f"  GSM problems found: {gsm_count}")
except Exception as e:
    print(f"Error: {e}")

# Check problems directory
print("\n" + "="*60)
print("CHECKING PROBLEMS DIRECTORY")
print("="*60)

problems_url = "https://api.github.com/repos/jettjaniak/chainscope/contents/chainscope/data/problems"
try:
    response = requests.get(problems_url)
    if response.status_code == 200:
        files = response.json()
        print(f"Found {len(files)} files in problems/:")
        
        # Look for GSM-related files
        for f in files[:10]:
            print(f"  - {f['name']}")
            if 'gsm' in f['name'].lower():
                print(f"    ^ GSM dataset found!")
                
                # Download to check size
                file_response = requests.get(f['download_url'])
                if file_response.status_code == 200:
                    if f['name'].endswith('.json'):
                        data = json.loads(file_response.text)
                        if isinstance(data, list):
                            print(f"    Contains {len(data)} GSM problems")
                            if data:
                                print(f"    Example: {data[0] if len(str(data[0])) < 200 else str(data[0])[:200] + '...'}")
                    elif f['name'].endswith('.yaml'):
                        data = yaml.safe_load(file_response.text)
                        if isinstance(data, dict):
                            print(f"    Contains {len(data)} GSM problems")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
ChainScope appears to focus on:
1. Implicit Post-Hoc Rationalization (comparing X>Y vs Y>X questions)
2. Restoration Errors (recovering from incorrect intermediate steps)
3. Unfaithful reasoning "in the wild" without prompting

Looking for GSM8K problems specifically...
""")