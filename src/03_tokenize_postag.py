import os
import spacy

def load_model(language):
    if language.lower() == "english":
        return spacy.load("en_core_web_sm")
    elif language.lower() == "dutch":
        return spacy.load("nl_core_news_md")
    else:
        raise ValueError(f"Unsupported language: {language}")

def pos_tag_text(text, language, nlp):
    doc = nlp(text)
    output_lines = []
    for sent in doc.sents:
        for token in sent:
            output_lines.append(f"{token.text}\t{token.pos_}")
        output_lines.append("")  # separate sentences
    return "\n".join(output_lines).strip()

def process_folder(input_folder, output_folder, language):
    os.makedirs(output_folder, exist_ok=True)
    nlp = load_model(language)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".txt"):
            input_file = os.path.join(input_folder, filename)
            with open(input_file, "r", encoding="utf-8") as f:
                text = f.read()
            tagged_text = pos_tag_text(text, language, nlp)
            output_file = os.path.join(output_folder, filename)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(tagged_text)
            print(f"Processed {filename}")

if __name__ == "__main__":
    subcorpora = {
        "PhilosophicalTransactions": ("../Antoni/English_E_LMod/PT_cleaned", "english"),
        "Letterbook": ("../Antoni/English_E_LMod/letterbook_translations", "english"),
        "Allebrieven": ("../Antoni/English_20th/Text", "english"),
        "Dutch": ("../Antoni/Original_Dutch/DutchTxt/Text", "dutch")
    }

    base_output_folder = "../Antoni_Corpus/"

    # process all subcorpora:
    # for label, (folder, language) in subcorpora.items():
    #     output_folder = os.path.join(base_output_folder, label)
    #     print(f"Processing folder {label} using the {language} model...")
    #     process_folder(folder, output_folder, language)

    # Process only the Dutch or another subcorpus
    label = "Dutch"
    folder, language = subcorpora[label]
    output_folder = os.path.join(base_output_folder, label)
    print(f"Processing folder {label} using the {language} model...")
    process_folder(folder, output_folder, language)
