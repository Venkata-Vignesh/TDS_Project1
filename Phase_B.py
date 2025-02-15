import os
import json
import sqlite3
import requests
import git
from PIL import Image
import markdown
from fastapi import FastAPI, Query
import pandas as pd
import whisper
from pathlib import Path
from bs4 import BeautifulSoup
import shutil

class SecurityError(Exception):
    """Custom exception for security violations"""
    pass

def validate_path(path):
    """
    Ensure all file operations are within /data directory
    Returns absolute path if valid, raises SecurityError if not
    """
    abs_path = os.path.abspath(path)
    data_dir = os.path.abspath("/data")
    
    if not abs_path.startswith(data_dir):
        raise SecurityError(f"Access denied: {path} is outside /data directory")
    return abs_path

def fetch_api_data(api_url, output_file):
    """B3: Fetch data from API and save it safely"""
    # Validate output path
    output_path = validate_path(output_file)
    
    # Fetch data
    response = requests.get(api_url)
    response.raise_for_status()
    
    # Save to validated path
    with open(output_path, 'w') as f:
        json.dump(response.json(), f, indent=2)

def handle_git_tasks(repo_url, local_path, commit_message):
    """B4: Clone repo and make commit safely"""
    # Validate repo path is within /data
    repo_path = validate_path(local_path)

    # Clone repo if it doesn't exist
    if not os.path.exists(repo_path):
        repo = git.Repo.clone_from(repo_url, repo_path)
    else:
        repo = git.Repo(repo_path)
    
    # Make changes and commit
    # Note: The actual changes would be made by other functions
    repo.index.add('*')  # This adds all changes in the repo
    repo.index.commit(commit_message)

def run_database_query(query, db_path, output_file):
    """B5: Run SQL query safely"""
    # Validate paths
    db_path = validate_path(db_path)
    output_path = validate_path(output_file)
    
    # Execute query
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn)
        df.to_csv(output_path, index=False)
    finally:
        conn.close()

def scrape_website(url, output_file):
    """B6: Scrape website data safely"""
    # Validate output path
    output_path = validate_path(output_file)
    
    # Fetch and parse webpage
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Save scraped data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'title': soup.title.string if soup.title else None,
            'text': soup.get_text(),
            'links': [a.get('href') for a in soup.find_all('a', href=True)]
        }, f, indent=2)

def process_image(input_file, output_file, max_size=(800, 800)):
    """B7: Compress/resize image safely"""
    # Validate paths
    input_path = validate_path(input_file)
    output_path = validate_path(output_file)
    
    # Process image
    with Image.open(input_path) as img:
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Resize maintaining aspect ratio
        img.thumbnail(max_size)
        
        # Save with compression
        img.save(output_path, 'JPEG', quality=85, optimize=True)

def transcribe_audio(input_file, output_file):
    """B8: Transcribe MP3 safely"""
    # Validate paths
    input_path = validate_path(input_file)
    output_path = validate_path(output_file)
    
    # Load model and transcribe
    model = whisper.load_model("base")
    result = model.transcribe(input_path)
    
    # Save transcription
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

def convert_markdown(input_file, output_file):
    """B9: Convert Markdown to HTML safely"""
    # Validate paths
    input_path = validate_path(input_file)
    output_path = validate_path(output_file)
    
    # Read and convert
    with open(input_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert to HTML
    html = markdown.markdown(md_content)
    
    # Save HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

app = FastAPI()

@app.get("/filter-csv")
async def filter_csv(
    csv_path: str,
    column: str,
    value: str = Query(...),
    operator: str = Query("gt", regex="^(eq|gt|lt|contains)$")
):
    """Filter CSV and return JSON data"""
    # Validate CSV path
    csv_path = validate_path(csv_path)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"DataFrame loaded: {df.head()}")  # Debugging line

    # Apply filter based on operator
    if operator == "eq":
        filtered_df = df[df[column] == value]
    elif operator == "gt":
        filtered_df = df[df[column] > float(value)]
    elif operator == "lt":
        filtered_df = df[df[column] < float(value)]
    elif operator == "contains":
        filtered_df = df[df[column].str.contains(value, na=False)]
    
    print(f"Filtered DataFrame: {filtered_df}")  # Debugging line
    return filtered_df.to_dict(orient='records')

# Bonus functionality: File monitoring
def setup_file_monitoring(directory):
    """Bonus: Monitor files in /data directory for changes"""
    watch_path = validate_path(directory)
    
    class FileChangeHandler:
        def __init__(self):
            self.last_modified = {}
            self.scan_files()
        
        def scan_files(self):
            for root, _, files in os.walk(watch_path):
                for file in files:
                    path = os.path.join(root, file)
                    mtime = os.path.getmtime(path)
                    if path not in self.last_modified or self.last_modified[path] != mtime:
                        self.last_modified[path] = mtime
                        self.handle_change(path)
        
        def handle_change(self, file_path):
            # Log change without deleting anything
            log_path = os.path.join(watch_path, 'file_changes.log')
            with open(log_path, 'a') as f:
                f.write(f"{file_path} modified at {pd.Timestamp.now()}\n")
    
    return FileChangeHandler()