import os
import json
from openai import OpenAI

# Initialize OpenAI client (set OPENAI_API_KEY in your environment, never hardcode it)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_files_dict(folder_path):
    files = {}
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            base = os.path.splitext(file)[0].lower()
            parts = base.split("_")
            if len(parts) < 2:
                continue
            refno = "_".join(parts[:-1])
            version = parts[-1]
            files.setdefault(refno, {})[version] = os.path.join(folder_path, file)
    return files

def read_text(file_path):
    if not file_path or not os.path.exists(file_path):
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def create_alignment_prompt(dutch, pt, letterbook, modern):
    return (
        "Below are four versions of the same letter. Some versions may be incomplete. "
        "Align these texts sentence by sentence so that each row corresponds to the same idea. "
        "Return the output as a JSON array of objects, where each object has the keys "
        "\"Dutch\", \"PT\", \"Letterbook\", and \"Modern\".\n\n"
        f"Dutch version:\n{dutch}\n\n"
        f"PT translation:\n{pt}\n\n"
        f"LetterBook translation:\n{letterbook}\n\n"
        f"Modern (English) translation:\n{modern}\n\n"
        "Return only valid JSON."
    )

def align_letter(prompt, model="gpt-4o", max_tokens=16384, temperature=0.2):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error calling GPT API:", e)
        return None

if __name__ == "__main__":
    folder_dutch = "../Antoni/Original_Dutch/DutchTxt/Text"
    folder_pt = "../Antoni/English_E_LMod/PT_cleaned"
    folder_letterbook = "../Antoni/English_E_LMod/letterbook_translations_renamed"
    folder_modern = "../Antoni/English_20th/Text"

    dutch_files = (get_files_dict(folder_dutch))
    pt_files = (get_files_dict(folder_pt))
    letterbook_files = (get_files_dict(folder_letterbook))
    modern_files = (get_files_dict(folder_modern))

    aggregated = {}

    for refno, versions in modern_files.items():
        modern_path = versions.get("english")
        if not modern_path:
            print(f"Skipping {refno}: Modern version missing.")
            continue

        dutch = read_text(dutch_files.get(refno, {}).get("dutch"))
        pt = read_text(pt_files.get(refno, {}).get("pt"))
        letterbook = read_text(letterbook_files.get(refno, {}).get("letterbook"))
        modern = read_text(modern_path)

        prompt = create_alignment_prompt(dutch, pt, letterbook, modern)
        print(f"=== RefNo: {refno} ===\n{prompt}\n{'='*80}\n")

        result = align_letter(prompt)
        if result:
            aggregated[refno] = result
        else:
            print(f"Alignment failed for {refno}")

    out_folder = "../Antoni_Corpus/AlignedOutput"
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, "antoni_alignment.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"Saved aggregated alignment output to {out_path}")


