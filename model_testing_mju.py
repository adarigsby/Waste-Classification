import os 
import subprocess 
from datasets import load_dataset

API_KEY = "RduP6OrDy86FMyWFrxtk"
MODEL_ID = "recyclable-waste-detection/1"
RESULTS_FILE = "mju_waste_classification_results.txt"

trashnet = load_dataset("garythung/trashnet", split="train")
image_folder_path = "mju-waste-v1.0/JPEGImages"

all_image_files = os.listdir(image_folder_path)
file_set = all_image_files[0:2475]

counter = 0

for image_file in file_set:
    image_file_path = os.path.join(image_folder_path, image_file)
    if os.path.isfile(image_file_path):
        waste_image = image_file_path

    image_data = { "image_info": image_file }

    command = [
        "inference", "infer", 
        "-i", waste_image, 
        "--api-key", API_KEY, 
        "--model_id", MODEL_ID
    ]

    try: 
        output = subprocess.run(command, capture_output=True, check=True, text=True,)
    except Exception as e:
        print(f"error processing the image: {e}")

    counter += 1
    print(f"counter: {counter}")

    with open(RESULTS_FILE, "a") as file:
        file.write(str(image_data))
        file.write("\n")
        file.write(output.stdout)
        file.write("\n")

print(f"{counter} images were processed")