from argparse import ArgumentParser
import pandas as pd
import os
from pathlib import Path
import requests
from PyPDF2 import PdfFileReader
from groq import Groq
from json_repair import repair_json


client = Groq(
    api_key=os.getenv(
        "GROQ_API_KEY"
    )
)


def predict(messages):
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=0,
        max_tokens=10240,
        top_p=0,
        stream=True,
        stop=None,
    )

    output = ""
    for chunk in completion:
        output += chunk.choices[0].delta.content or ""
    return {"output": output}


def process(src_path):
    src_path = Path(src_path)
    resume_dir = src_path.parent / src_path.stem + "-resumes"
    resume_dir.mkdir(exist_ok=True, parents=True)
    output_path = src_path.parent / src_path.stem + "-processed.csv"
    df = pd.read_csv(src_path)

    for idx, row in df.iterrows():
        try:
            resume_link = row["Resume"]
            id = resume_link.split("id=")[1]
            dest_file_location = resume_dir / (id + ".pdf")
            if not dest_file_location.exists():
                # download the file
                download_link = f"https://drive.google.com/uc?id={id}"
                print(f"Downloading {download_link} to {dest_file_location}")
                response = requests.get(download_link)
                with open(dest_file_location, "wb") as f:
                    f.write(response.content)

            reader = PdfFileReader(str(dest_file_location))
            text = ""
            for page_no, page in enumerate(reader.pages):
                text += page.extract_text()
            assert (
                len(text) < 6000
            )  # ignore if the resume is large as it might error out any wya because of token limit
            assert text, f"Empty text for {dest_file_location}"
            messages = [
                {
                    "role": "system",
                    "content": "Extract college information where the candidate has finished and the grduating year of his Bachelors/engineering based on the text extracted from the pdf file of a resume, pick only the latest college, do not include any addition info in the response, just the college name so that I can store the output in DB and use it for further processing. Give the output as a json document which contains fields 'college', 'degree', 'graduating_year'. If the college name is not present in the text, return an empty string.",
                },
                {"role": "user", "content": f"here is the extracted text: {text}"},
            ]
            response = predict(messages)
            print(response, id)
            clean_json = repair_json(response["output"], return_objects=True)
            df.at[idx, "college"] = clean_json.get("college", "")
            df.at[idx, "degree"] = clean_json.get("degree", "")
            df.at[idx, "graduating_year"] = clean_json.get("graduating_year", "")
            print(
                df["college"].iloc[idx],
                df["degree"].iloc[idx],
                df["graduating_year"].iloc[idx],
            )
        except Exception as e:
            print(f"Error for {id}: {e}")

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "src_path",
        type=str,
        deescription="Path to the csv file where the resume metadata is available",
    )
    args = parser.parse_args()
    src_path = args.src_path
    process(src_path)

