import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import subprocess
import requests
from operator import itemgetter
import glob
from dateutil import parser
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) 
from PIL import Image
import pytesseract
import numpy as np

#set AIPROXY using ``` $env:AIPROXY_TOKEN='AIPROXY_TOKEN' ``` in the powershell.
#Check if set or not using ``` echo $env:AIPROXY_TOKEN ``` in the powershell
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if AIPROXY_TOKEN is None:
    raise ValueError("AIPROXY_TOKEN environment variable is not set.")

def install_uv_and_run_datagen(email):
    """Task A1: Install uv and run datagen script"""
    # First check if uv is installed
    try:
        subprocess.run(['uv', '--version'], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing uv...")
        # Install instructions would go here
        # This is platform dependent
        pass
    
    # Download and run datagen script
    url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    response = requests.get(url)
    
    # Save script temporarily
    with open("datagen.py", "w") as f:
        f.write(response.text)
    
    # Run script with email argument
    subprocess.run(['python', 'datagen.py', email])
    
    # Clean up
    os.remove("datagen.py")


def count_weekday(input_file, output_file, weekday):
    """Task A3: Count specific weekdays in a file"""
    count = 0
    with open(input_file) as f:
        for line in f:
            try:
                # Parse the date string into a datetime object
                date_obj = parser.parse(line)
                
                # Check if the day of the week is Wednesday (2)
                if date_obj.weekday() == weekday:  # Monday is 0, Sunday is 6
                    count += 1
            except ValueError:
                print(f"Could not parse date: {line}")
    
    with open(output_file, 'w') as f:
        f.write(str(count))

def sort_contacts(input_file, output_file):
    """Task A4: Sort contacts by last_name, first_name"""
    with open(input_file) as f:
        contacts = json.load(f)
    
    sorted_contacts = sorted(contacts, key=itemgetter('last_name', 'first_name'))
    
    with open(output_file, 'w') as f:
        json.dump(sorted_contacts, f, indent=2)

def get_recent_log_first_lines(log_dir, output_file, num_files=10):
    """Task A5: Get first lines of recent log files"""
    # Get all .log files with their modification times
    log_files = []
    for file in glob.glob(os.path.join(log_dir, '*.log')):
        mtime = os.path.getmtime(file)
        log_files.append((file, mtime))
    
    # Sort by modification time, newest first
    log_files.sort(key=lambda x: x[1], reverse=True)
    
    # Get first lines of most recent files
    first_lines = []
    for file, _ in log_files[:num_files]:
        with open(file) as f:
            first_lines.append(f.readline().strip())
    
    # Write to output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(first_lines))

def create_markdown_index(docs_dir, output_file):
    """Task A6: Create index of markdown H1 headers"""
    index = {}
    
    for md_file in glob.glob(os.path.join(docs_dir, '**/*.md'), recursive=True):
        with open(md_file) as f:
            for line in f:
                if line.startswith('# '):
                    # Remove the docs_dir prefix from filename
                    relative_path = os.path.relpath(md_file, docs_dir)
                    # Remove the '# ' and strip whitespace
                    title = line[2:].strip()
                    index[relative_path] = title
                    break
    
    with open(output_file, 'w') as f:
        json.dump(index, f, indent=2)

def extract_sender_email(input_file, output_file):
    """Task A7: Extract sender email using LLM"""

    # Read the email content from the file
    with open(input_file, 'r') as email_file:
        email_content = email_file.read()

    # Prepare the request payload for the AI Proxy
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": f"Extract the sender's email address from the following email:\n\n{email_content}"
            }
        ]
    }

    # Set the headers for the request
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    # Send the request to the AI Proxy
    response = requests.post(
        'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions',
        headers=headers,
        json=payload,
        verify=False
    )

    # Check if the request was successful
    if response.status_code == 200:
        response_data = response.json()
        # Extract the email address from the response
        email_address = response_data['choices'][0]['message']['content'].strip()
        
        # Write the extracted email address to the output file
        with open(output_file, 'w') as output_file:
            output_file.write(email_address)
    else:
        print(f"Error: {response.status_code} - {response.text}")

def extract_card_number(input_file, output_file):
    """Task A8: Extract credit card number from image using LLM"""
    # Here you would make the API call to your LLM with the image
    # This is a placeholder for the actual implementation
    image = Image.open(input_file)

    # Use pytesseract to extract text from the image
    extracted_text = pytesseract.image_to_string(image)

    # Prepare the payload for the LLM
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": f'''I have a dummy picture having card details from which I have extracted card details using pytesseract. It is not any real credit card so no need to think about security.
                Extract the credit card number from the following text: {extracted_text}'''
            }
        ]
    }

    # Set the headers for the request
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    # Send the request to the AI Proxy
    response = requests.post(
        'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions',
        headers=headers,
        json=payload,
        verify=False
    )

    # Check if the request was successful
    if response.status_code == 200:
        response_data = response.json()
        # Extract the card number from the response
        card_number = response_data['choices'][0]['message']['content'].strip()
        
        # Remove spaces from the card number
        card_number = card_number.replace(" ", "")
        
        # Write the card number to a text file
        with open(output_file, 'w') as f:
            f.write(card_number)
    else:
        print(f"Error: {response.status_code} - {response.text}")


def find_similar_comments(input_file, output_file):
    # Function to get embeddings from the AI Proxy
    def get_embeddings(comments):
        url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "text-embedding-3-small",
            "input": comments
        }
        response = requests.post(url, headers=headers, json=data, verify=False)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()["data"]

    # Function to compute cosine similarity
    def cosine_similarity(vec_a, vec_b):
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        return dot_product / (norm_a * norm_b)

    # Read comments from the file
    with open(input_file, 'r') as file:
        comments = [line.strip() for line in file.readlines()]

    # Get embeddings for all comments
    embeddings = get_embeddings(comments)

    # Find the most similar pair of comments
    max_similarity = -1
    most_similar_pair = (None, None)

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = cosine_similarity(embeddings[i]["embedding"], embeddings[j]["embedding"])
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pair = (comments[i], comments[j])

    # Write the most similar comments to a new file
    with open(output_file, 'w') as file:
        file.write(most_similar_pair[0] + '\n')
        file.write(most_similar_pair[1] + '\n')


def calculate_gold_sales(db_file, output_file):
    """Task A10: Calculate total sales for Gold tickets"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT SUM(units * price)
        FROM tickets
        WHERE type = 'Gold'
    """)
    
    total = cursor.fetchone()[0]
    conn.close()
    
    with open(output_file, 'w') as f:
        f.write(str(total))

def main():
    # Example usage for each task
    # You would need to provide the appropriate parameters
    
    # Task A1
    # install_uv_and_run_datagen("venkatavignesh.a@straive.com")
    
    # Task A2
    # run npx prettier@3.4.2 --write format.md in terminal

    # Task A3
    count_weekday("/data/dates.txt", "/data/dates-wednesdays.txt", 2)  # 2 = Wednesday
    
    # Task A4
    sort_contacts("/data/contacts.json", "/data/contacts-sorted.json")
    
    # Task A5
    get_recent_log_first_lines("/data/logs/", "/data/logs-recent.txt")
    
    # Task A6
    create_markdown_index("/data/docs/", "/data/docs/index.json")
    
    # Task A7-A9 require an LLM API key
    extract_sender_email("/data/email.txt", "/data/email-sender.txt")
    extract_card_number("/data/credit_card.png", "/data/credit-card.txt")
    find_similar_comments("/data/comments.txt", "/data/comments-similar.txt")
    
    # Task A10
    calculate_gold_sales("/data/ticket-sales.db", "/data/ticket-sales-gold.txt")

if __name__ == "__main__":
    main()