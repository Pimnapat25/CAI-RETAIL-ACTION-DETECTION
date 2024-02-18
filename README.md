# CAI-RETAIL HACKATHON-ACTION DETECTION
An action detection model based on TSN (Time-Sensity Networking) from [MMAction2](https://github.com/open-mmlab/mmaction2). in OpenMMLab Project.
## 🎯Description
The model has been trained extensively on diverse datasets to specialize in **real-time activity recognition** within a convenience store. It possesses the capability to determine **the position of the POS** in which action is occurring as well as precisely identify **the start and end times** of detected actions. This ensures a detailed and timely analysis of activities occurring in the store.

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

### Step 1: Install OpenMIM, MMEngine, and MMCV
```bash
%pip install -U openmim
!mim install mmengine
!mim install "mmcv>=2.0.0"
```
### Step 2 : Clone and Install MMAction2
```bash
!rm -rf mmaction2
!git clone https://github.com/open-mmlab/mmaction2.git -b main
%cd mmaction2
!pip install -e .
```
### Step 3 : Install Optional Requirements and Timm
```bash
!pip install -r requirements/optional.txt
!pip install timm
```

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

for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        # Extract the filename without extension
        filename = os.path.splitext(video_file)[0]

        # Set the path for the JSON output file
        json_output_file = os.path.join("/path/to/your/output/folder/JSON", f"{filename}.json") #<----- Here's

        # Run the action detection script for each video file
       command = f"python /path/to/mmaction2/demo/long_video_demo.py " \
                  f"/path/to/config.py " \
                  f"/path/to/action_recognition_model.pth " \
                  f"\"{os.path.join(video_folder, video_file)}\" " \
                  f"/path/to/class_list.txt " \
                  f"\"{json_output_file}\" --input-step 25 --stride 1"

        os.system(command)
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
json_folder = "/path/to/your/output/folder/JSON", f"{filename}.json" #<----- Here's

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
import os
import json
import csv

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
json_folder = "/path/to/your/output/folder/JSON" #<----- Here's
output_csv = "/path/to/your/output/folder/JSON/convert_into_CSV" #<----- Here's

# Open CSV file for writing
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['Clipname', 'Action', 'StartTime', 'StopTime', 'POS_x', 'POS_y', 'Runtime(sec)'])

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
                # Write row to CSV file
                writer.writerow([name, action, start_frame, stop_frame])

print("CSV file generated successfully.")
```
_Make sure to replace "/path/to/your/output/folder/JSON", "/path/to/your/output/folder/JSON/convert_into_CSV" with the actual file paths_

### Step 5 : Results
After all process is done you should get an output as .csv file.\
"output example here"

## ✒️Author and acknowledgement
- [Your Name]
- Email: your.email@example.com
- GitHub: [Your GitHub Profile](https://github.com/your-username)

We would like to express our gratitude to the following individuals and projects that contributed to the development of this project:

- [Name of Contributor 1](https://github.com/contributor1)
- [Name of Contributor 2](https://github.com/contributor2)



