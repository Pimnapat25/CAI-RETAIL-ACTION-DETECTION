# CAI-RETAIL HACKATHON-ACTION DETECTION
An action detection model based on TSN (Time-Sensity Networking) from [MMAction2](https://github.com/open-mmlab/mmaction2). in OpenMMLab Project.
## 🎯Description
The model has been trained extensively on diverse datasets to specialize in **activity recognition** within a convenience store. It possesses the capability to determine **the position of the POS** in which action is occurring as well as identify **the start and end times** of detected actions. This ensures a detailed and timely analysis of activities occurring in the store.

🖊️The actions that can be recognized are as follows 🖊️:
- 01 product purchasing
* 02 bill payment
+ 05 sales recording in the pos system
- 06 cash payment with change
* 07 cash payment without change
+ 08 true money wallet payment
- 09 credit card payment
* 10 automatic payment kiosk purchasing
+ 12 TrueMoney Wallet refilling without depositing the money into the cash register
- 13 post-transaction bill editing
## 📥Installation
To set up the necessary dependencies for your project, follow the steps below:

### Install OpenMIM, MMEngine, and MMCV
```bash
%pip install -U openmim
!mim install mmengine
!mim install "mmcv>=2.0.0"
```
### Clone and Install MMAction2
```bash
!rm -rf mmaction2
!git clone https://github.com/open-mmlab/mmaction2.git -b main
%cd mmaction2
!pip install -e .
```
### Install Optional Requirements and Timm
```bash
!pip install -r requirements/optional.txt
!pip install timm
```
## 🙂Model
Download _class_list.txt_ file [here](ClassList_final.txt)

Download _config.py_ file [here](config.py)

Download _action_recognition_model_ [here](https://drive.google.com/file/d/111y6QjIo78JjUf--j5ud4yxrKyzIfD5e/view?usp=sharing)
## 📂Requirements
- Python 3.7 or higher
* CUDA 10.2 or higher
+ PyTorch 1.8 or higher
  
## 🖥️Usage
### Step 1 : Set Video Folder Path
Set the path to the folder containing videos you want to execute
```bash
video_folder = "/path/to/your/video/folder"
```
### Step 2 : Run Action Recognition Script
Run the following command in the root directory
```bash
import os
import time
import csv

# Set the path to the folder containing the videos
video_folder = "/content/drive/MyDrive/CAI_Hackathon/evaluate" #<----- Here's

runtime_csv = "/content/runtime.csv"
runtime_dir = os.path.dirname(runtime_csv)

with open(runtime_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['filename','Runtime (Sec)'])
    # Iterate over each video file in the folder
    for video_file in os.listdir(video_folder):
        if video_file.endswith(".mp4"):
            # Extract the filename without extension
            filename = os.path.splitext(video_file)[0]

            # Set the path for the JSON output file
            json_output_file = os.path.join("/content/drive/MyDrive/CAI_Hackathon/ModelCAI/ResultJson", f"{filename}.json") #<----- Here's

            start_time = time.time()

            # Run the action detection script for each video file
            command = f"python /content/mmaction2/demo/long_video_demo.py " \
                      f"/content/drive/MyDrive/ModelCAI/CAI_Hackathon/TSN_RES101/Res101_W20240210_Acc605/20240210_042345/vis_data/config.py " \
                      f"/content/drive/MyDrive/ModelCAI/CAI_Hackathon/TSN_RES101/Res101_W20240210_Acc605/best_acc_top1_epoch_50.pth " \
                      f"\"{os.path.join(video_folder, video_file)}\" " \
                      f"/content/drive/MyDrive/CAI_Hackathon/ModelCAI/ClassList_final.txt " \
                      f"\"{json_output_file}\" --input-step 25 --stride 1"

            os.system(command)
            end_time = time.time()

            runtime = end_time - start_time
            writer.writerow([filename,runtime])
```
_Make sure to replace "/path/to/your/video/folder", "/path/to/your/output/folder/JSON", "/path/to/config.py", "/path/to/action_recognition_model.pth", and "/path/to/class_list.txt" with the actual file paths_

### Step 3 : Make Result Using JSON 
```bash
import os
import json

def predict_action_segments_from_json(json_file):
    action_segments = []
    current_action = None
    start_time = None

    with open(json_file, 'r') as f:
        json_data = json.load(f)

        for frame_number, frame_data in json_data.items():
            if isinstance(frame_data, dict):
                if '1' in frame_data:
                    action_label = frame_data['1'].split(': ')[0].strip()
                    if action_label != current_action:
                        if current_action is not None:
                            end_time = int(frame_number) - 1
                            action_segments.append((current_action, start_time, end_time))
                        current_action = action_label
                        start_time = int(frame_number)
            else:
                current_action = None
                start_time = None

    if current_action is not None:
        action_segments.append((current_action, start_time, int(frame_number)))

    return action_segments

# Set the path to the folder containing the JSON files
json_folder = "/content/drive/MyDrive/CAI_Hackathon/ModelCAI/ResultJson" #<----- Here's

# Loop over each JSON file in the folder
for json_file_name in os.listdir(json_folder):
    if json_file_name.endswith(".json"):
        json_file_path = os.path.join(json_folder, json_file_name)
        action_segments = predict_action_segments_from_json(json_file_path)

        for action, start_time, end_time in action_segments:
            # Extract name from file name
            name = os.path.splitext(json_file_name)[0]
            # Convert start and stop times to frames
            start_frame = (start_time - 25) // 25
            stop_frame = (end_time - 25) // 25
            # Print the segment in the desired format
            print(f"{name} {action} {start_frame} {stop_frame}")
```
_Make sure to replace "/path/to/your/output/folder/JSON" with the actual file path_

### Step 4 : Convert JSON to CSV
```bash
from typing_extensions import final
import os
import json
import csv
import cv2
import time

def extract_action_label(raw_action_label):
    return raw_action_label.split('=')[1].split(':')[0].strip()

def predict_action_segments_from_json(json_file):
    action_segments = []
    current_action = None
    start_time = None

    with open(json_file, 'r') as f:
        json_data = json.load(f)

        for frame_number, frame_data in json_data.items():
            if isinstance(frame_data, dict):
                if '1' in frame_data:
                    # Extract the action label using the extract_action_label function
                    action_label = extract_action_label(frame_data['1'])
                    if action_label != current_action:
                        if current_action is not None:
                            end_time = int(frame_number) - 1
                            action_segments.append((current_action, start_time, end_time))
                        current_action = action_label
                        start_time = int(frame_number)
            else:
                current_action = None
                start_time = None

    if current_action is not None:
        action_segments.append((current_action, start_time, int(frame_number)))

    return action_segments



# Set the path to the folder containing the JSON files
json_folder = "/content/drive/MyDrive/CAI_Hackathon/ModelCAI/ResultJson" #<----- Here's
output_csv = "/content/drive/MyDrive/CAI_Hackathon/ModelCAI/ResultJson/output.csv" #<----- Here's

model = YOLO("/content/drive/MyDrive/CAI_Hackathon/final_detectWeight.pt")

output_csv = "/content/output.csv"
output_dir = os.path.dirname(output_csv)

# Open CSV file for writing
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(["id",'ClipName', 'ActionNumber', 'StartTime', 'EndTime'])
    count = 0
    # Iterate over each JSON file in the folder
    for json_file_name in os.listdir(json_folder):
        if json_file_name.endswith(".json"):
            json_file_path = os.path.join(json_folder, json_file_name)
            action_segments = predict_action_segments_from_json(json_file_path)
            # Extract name from file name
            name = os.path.splitext(json_file_name)[0]
            for action, start_time, end_time in action_segments:
                # Convert start and stop times to frames
                start_frame = (start_time - 25) // 25
                stop_frame = (end_time - 25) // 25
                count = count + 1
                # Write row to CSV file
                writer.writerow([count,name, action, start_frame, stop_frame])

time_csv = "/content/time.csv"
time_dir = os.path.dirname(time_csv)

with open(time_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['Image_name','POS_x', 'POS_y'])
    pos_x, pos_y = 0, 0
    for img_file_name in os.listdir(evaluate_folder_path):
      start_time = time.time()
      img_file_path = os.path.join(evaluate_folder_path, img_file_name)
      img = cv2.imread(img_file_path)
      results = model.predict(img)

      name = os.path.splitext(img_file_name)[0]

      print(img_file_path,"--> complete")
      if len(results[0].boxes) == 0:
          end_time = time.time()
          writer.writerow([name, 0, 0])
      else:
          for box in results[0].boxes:
            class_id = box.cls.item()
            if class_id == 0:
              x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
              pos_x_center = (x1 + x2) / 2  # Center X coordinate
              pos_y_center = (y1 + y2) / 2  # Center Y coordinate
              end_time = time.time()
              writer.writerow([name,pos_x_center, pos_y_center])
            elif class_id != 0:
              end_time = time.time()
              writer.writerow([name,0, 0])

import pandas as pd

# Read the CSV files into dataframes
df1 = pd.read_csv('/content/output.csv')
df2 = pd.read_csv('/content/time.csv')
df3 = pd.read_csv('/content/runtime.csv')

# Merge the dataframes based on the 'Clipname' and 'Image_name' columns
merged_df = pd.merge(df1, df2, left_on='ClipName', right_on='Image_name')
merged_df_2 = pd.merge(merged_df, df3, left_on='ClipName', right_on='filename')

num_rows = merged_df_2.shape[0]

for i in range(1, num_rows + 1):
  merged_df_2.loc[i - 1, 'id'] = i

final = merged_df_2.drop('Image_name', axis = 1)

# Write the merged dataframe to a new CSV file
final.to_csv('/content/final.csv', index=False)

print("CSV file generated successfully.")
```
_Make sure to replace "/path/to/your/output/folder/JSON", "/path/to/your/output/folder/JSON/convert_into_CSV" with the actual file paths_

### Step 5 : Results
After all processes are done you should get an output as .csv file. Here is the example!

![Output](https://github.com/Pimnapat25/CAI-RETAIL-ACTION-DETECTION/assets/112639020/5ef2f844-1adf-4025-ac29-d6d978b59861)


## ✒️Author and acknowledgement
1. Kasidit Prajongkarn
   - Email: totti.kasidit@gmail.com
2. Phatthadon Kamnasak
   - Email: phatthadon.kam@gmail.com
3. Pachara Sapbamrer
   - Email: pacharapor18793@gmail.com
4. Pimnapat Koovuthyakorn
   - Email: pimnapatkoov@gmail.com
     
We would like to express our gratitude to the following individuals and projects that contributed to the development of this project:

- MMAction2 (https://github.com/open-mmlab/mmaction2/blob/main/README.md)
- คุณพีรเดช บางเจริญทรัพย์, Machine Learning Engineer at TikTok (Singapore)
- CPALL Company



