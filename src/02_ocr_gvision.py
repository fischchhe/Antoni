import os
import re
from google.cloud import vision


def extract_refno(filename):
#extract the first 3 parts which are sperated by and underscore
    parts = filename.split('_')
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return None


def extract_page_number(filename):
    match = re.search(r'page(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def group_files_by_refno(folder):
#dict creation
    ref_groups = {}
    for file in os.listdir(folder):
        if file.lower().endswith(".png"):
            ref = extract_refno(file)
            if ref:
                abs_path = os.path.abspath(os.path.join(folder, file))
                ref_groups.setdefault(ref, []).append(abs_path)
    #  sorts files by page number per group
    for ref, files in ref_groups.items():
        files.sort(key=lambda path: extract_page_number(os.path.basename(path)))
    return ref_groups


def ocr_image(image_path, client, language_hints=['eng'], detection_type='text'):
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # image context creationm with lanaugge hints
    image_context = vision.ImageContext(language_hints=language_hints)

    if detection_type == 'document':
        response = client.document_text_detection(image=image, image_context=image_context)
    else:
        response = client.text_detection(image=image, image_context=image_context)

    if response.error.message:
        raise Exception(f"Error processing {image_path}: {response.error.message}")
    return response.full_text_annotation.text


def process_folder(input_folder, output_folder):

    client = vision.ImageAnnotatorClient()
    groups = group_files_by_refno(input_folder)

    # Create output folder if its not there yet
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for ref, file_paths in groups.items():
        print(f"Processing reference {ref} with {len(file_paths)} pages...")
        combined_text = ""
        for file_path in file_paths:
            try:
                text = ocr_image(file_path, client, language_hints=['eng'], detection_type='document')
                combined_text += text + "\n"
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        # Write the text to a file named by the reference number
        output_file = os.path.join(output_folder, f"{ref}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(combined_text)
        print(f"Written output for {ref} to {os.path.abspath(output_file)}")


if __name__ == "__main__":
    input_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Antoni", "English_E_LMod", "Philtrans")
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Antoni", "English_E_LMod", "Philtrans_vision_output")
    process_folder(input_folder, output_folder)
