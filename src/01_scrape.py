import pandas as pd
import requests
import os


#Define how the url starts
base_url = "https://rs-website-files-prod.s3.eu-west-2.amazonaws.com/pdf-cache/"

# tell p where to find the list of names
df = pd.read_excel("download_records.xlsx")

# tell p where to save the pdfs
os.makedirs("downloaded_pdfs", exist_ok=True)

#downloading
for file_name in df["RefNo"]:
    pdf_url = f"{base_url}{file_name}.pdf"  # Add the .pdf extension
    output_path = os.path.join("downloaded_pdfs", f"{file_name}.pdf")

    try:
        print(f"Downloading {file_name}...")
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise error for HTTP issues

        # Save the PDF
        with open(output_path, "wb") as pdf_file:
            pdf_file.write(response.content)

        print(f"Downloaded: {file_name}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {file_name}: {e}")

print("downloads complete")