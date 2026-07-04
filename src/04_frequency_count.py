import os
import re
from collections import Counter
import pandas as pd
import string

#this is just counting the frequewncies of firt-person pronouns

def read_token_lines(filepath):
    with open(filepath, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def split_paragraphs(token_lines):
    paragraphs = []
    paragraph = []
    for line in token_lines:
        if line == "SPACE":
            if paragraph:
                paragraphs.append(paragraph)
                paragraph = []
        else:
            paragraph.append(line)
    if paragraph:
        paragraphs.append(paragraph)
    return paragraphs

def count_pronouns(paragraphs, lang):
    pronouns = DUTCH_PRONOUNS if lang == "Dutch" else ENGLISH_PRONOUNS
    table = str.maketrans('', '', string.punctuation)

    flat_tokens = [
        line.split()[0].lower().translate(table)  # lowercase + strip punctuation
        for para in paragraphs for line in para if line != "SPACE"
    ]
    total_tokens = len(flat_tokens)
    counts = Counter(token for token in flat_tokens if token in pronouns)
    return sum(counts.values()), total_tokens


def process_subcorpus(label, folder, exclude_paragraphs=True):
    lang = "Dutch" if label == "Dutch" else "English"
    rows = []

    for filename in os.listdir(folder):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(folder, filename)
        token_lines = read_token_lines(filepath)
        paragraphs = split_paragraphs(token_lines)

        if exclude_paragraphs and len(paragraphs) > 6:
            paragraphs = paragraphs[3:-3]

        pronoun_count, total_tokens = count_pronouns(paragraphs, lang)
        norm_freq = (pronoun_count / total_tokens) * 1000 if total_tokens > 0 else 0

        rows.append({
            "refno": os.path.splitext(filename)[0],
            "subcorpus": label,
            "pronoun_count": pronoun_count,
            "token_count": total_tokens,
            "normalized_freq": norm_freq
        })

    return rows

# list of pronouns thats counted, i tried to cover as mcuh gorund as possible 

DUTCH_PRONOUNS = {
    "ik", "mij", "me", "mijn", "mijzelf", "mezelf",
    "wij", "we", "ons", "onze", "onszelf",

    "mijfe",       
    "mijfelv",
    "mefelv",      
    "meyn",     
    "mijnfelv", 
    "wijfelv",     
    "onf",      
    "onfe",        
    "onfelv",   
    "onfelve",  
    "onse",        
    "onzen",       
    "uyt onfen",   
    "wy",          
    "wÿ",       
    "onsefelven",  
    "onselven",    
}


ENGLISH_PRONOUNS = {
    "i", "me", "my", "mine", "myself", "myne",
    "we", "wee", "us", "vs", "our", "ovr", "oure", "ours", "ourselves",


    "myſelf", "myſelue", "myneſelf",
    "ourſelves", "ourſelues", "ourſelf",
    "vsſelves", "vſſelves", "vſ",
    "ouf", "oufſelves", "ourfelf", "ourfelues", "myfelf", "myfelue",
    "ovrfelves", "vf",
}


SUBCORPORA = {
    "Dutch": "../Antoni_Corpus/Dutch",
    "Allebrieven": "../Antoni_Corpus/Allebrieven",
    "Letterbook": "../Antoni_Corpus/letterbook_renamed",
    "PT": "../Antoni_Corpus/PhilosophicalTransactions"
}




if __name__ == "__main__":
    all_data = []

    for label, folder_path in SUBCORPORA.items():
        exclude = label != "PT"
        print(f"Processing {label} (exclude first & last 3 paras: {exclude})")
        subcorpus_rows = process_subcorpus(label, folder_path, exclude_paragraphs=exclude)
        all_data.extend(subcorpus_rows)

    df = pd.DataFrame(all_data)
    print(df.head())
    df.to_csv("first_person_pronoun_frequencies.csv", index=False)
