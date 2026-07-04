import os
import json
from openai import OpenAI

# Initialize client (set OPENAI_API_KEY in your environment, never hardcode it)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def read_text(file_path):
    if not file_path or not os.path.exists(file_path):
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def create_alignment_prompt(dutch, pt, letterbook, modern):
    prompt = (
        "Below are four versions of the same letter. Some versions may be incomplete. "
        "Align these texts sentence by sentence so that each row corresponds to the same idea. "
        "Return the output as a JSON array of objects, where each object has the keys "
        "\"Dutch\", \"PT\", \"Letterbook\", and \"Modern\".\n\n"
        f"Dutch version:\n{dutch}\n\n"
        f"PT translation:\n{pt}\n\n"
        f"Letterbook translation:\n{letterbook}\n\n"
        f"Modern (English) translation:\n{modern}\n\n"
        "Return only valid JSON."
    )
    return prompt


def align_letter(prompt, model="gpt-4o", max_tokens=16384, temperature=0.2):
    """Calls the OpenAI API to align the texts based on the provided prompt."""
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
    dutch_path = "../Antoni/Original_Dutch/DutchTxt/Text/el_l3_84_dutch.txt"
    pt_path = "../Antoni/English_E_LMod/PT_cleaned/el_l3_84_PT.txt"
    letterbook_path = "../Antoni/English_E_LMod/letterbook_translations_renamed/el_l3_84_LetterBook.txt"
    modern_path = "../Antoni/English_20th/Text/el_l3_84_english.txt"

    dutch_text = read_text(dutch_path)
    pt_text = read_text(pt_path)
    letterbook_text = read_text(letterbook_path)
    modern_text = read_text(modern_path)

    prompt = create_alignment_prompt(dutch_text, pt_text, letterbook_text, modern_text)
    print("=== Alignment Prompt ===\n", prompt, "\n" + "=" * 80 + "\n")

# call gpt api
    alignment_result = align_letter(prompt)

    if alignment_result:
        aggregated = {"el_l3_84": alignment_result}
        output_folder = "../Antoni_Corpus/AlignedOutput_el_l3_84"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "antoni_alignment.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, ensure_ascii=False, indent=2)
        print(f"Saved alignment output to {output_path}")
    else:
        print("Alignment failed for the single letter.")

