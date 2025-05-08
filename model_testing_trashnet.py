import os 
import subprocess 
from datasets import load_dataset

API_KEY = "RduP6OrDy86FMyWFrxtk"
MODEL_ID = "recyclable-waste-detection/1"
RESULTS_FILE = "trashnet_waste_classification_results.txt"
IMAGE_JPG = "dataset_image.jpg"

trashnet = load_dataset("garythung/trashnet", split="train")

for i in range(0,5054):
    waste_image = trashnet[i].get("image", None)
    label = trashnet[i]["label"]
    image_data = {
        "image_info": str(waste_image), 
        "actual_class_id": label
    }

    waste_image.save(IMAGE_JPG)

    command = [
        "inference", "infer", 
        "-i", IMAGE_JPG, 
        "--api-key", API_KEY, 
        "--model_id", MODEL_ID
    ]

    try: 
        output = subprocess.run(command, capture_output=True, check=True, text=True,)
    except Exception as e:
        print(f"error processing the image: {e}")

    with open(RESULTS_FILE, "a") as file:
        file.write(str(image_data))
        file.write("\n")
        file.write(output.stdout)
        file.write("\n")
