# -*- coding: utf-8 -*-
"""
Batch dedup JSON files by semantic similarity of {PICO, llm_text} using OpenAI embeddings.
Keep the newest year and higher GRADE when duplicates exist.

Outputs:
  output_dir/
    unique/      # kept jsons (same relative structure)
    duplicates/  # removed jsons
    dedup_report.jsonl  # mapping report
"""

import os
import json
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from config import Config
import numpy as np
from tqdm import tqdm

raw_path = '/data_vision9/ywang/PICO3rd_1210'
process_path = '/data_vision9/ywang/heisesuliuMVP_final_1209_processed_dedeup0_99/unique'
output_path = '/data_vision9/ywang/heisesuliuMVP_final_1209_processed_dedeup0_99/unique'
process_path = process_path + '/'
json_files = [f for f in os.listdir(process_path) if f.endswith('_pico.json')]

for json_file in tqdm(json_files):
    json_file_path = os.path.join(process_path, json_file)
    raw_json_file_path = os.path.join(raw_path, json_file)
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    with open(raw_json_file_path, 'r') as f:
        raw_json_data = json.load(f)
    
    for item in json_data:
        item['reference'] = raw_json_data[0]['reference']
        # item['year'] = raw_json_data[0]['year']
        # item['grade'] = raw_json_data[0]['GRADE']
        # item['country'] = raw_json_data[0]['country']
        # item['guideline_focus'] = raw_json_data[0].get('guideline_focus', 'treatment')
    json_file_outputpath = os.path.join(output_path, json_file)
    with open(json_file_outputpath, 'w',encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)