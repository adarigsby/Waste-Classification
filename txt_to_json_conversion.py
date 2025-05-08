import json 

def convert(txt_file, json_file):

    all_data = []
    lines = []

    with open(txt_file, "r") as file:
        for line in file:
            line.strip()
            lines.append(line)

    image_data = [lines[i:i+4] for i in range(0, len(lines), 4)]

    for single_image_data in image_data:
        data = {}
        try: 
            image_info_line = single_image_data[0].replace("'", '"')
            inference_result_line = single_image_data[2].replace("'", '"')

            if txt_file != "mju_waste_classification_results.txt":
                data["actual_class_id"] = json.loads(image_info_line).get("actual_class_id")

            if txt_file == "resort_it_waste_classification_results.txt":
                class_id_map = {1: 2, 2: 3, 3: 4, 4: 6}
                if int(data["actual_class_id"]) in class_id_map:
                    data["actual_class_id"] = class_id_map[int(data["actual_class_id"])]

            data["inference_id"] = json.loads(inference_result_line).get("inference_id")
            data["time"] = json.loads(inference_result_line).get("time")
            data["image"] = json.loads(inference_result_line).get("image")
            data["predictions"] = json.loads(inference_result_line).get("predictions", [])

            all_data.append(data)

        except Exception as e: 
            print(f"error: {e}")

    with open(json_file, "w") as file:
        json.dump(all_data, file, indent=4)

convert("trashnet_waste_classification_results.txt", "trashnet_waste_classification_results.json")
convert("mju_waste_classification_results.txt", "mju_waste_classification_results.json")
convert("resort_it_waste_classification_results.txt", "resort_it_waste_classification_results.json")
