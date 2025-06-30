# run_training.py
import sys
import os
import subprocess

# --- Configuration ---
# This is the only line you might ever need to change.
# It points to the yolov5 folder SIBLING to your project folder.
YOLOV5_PATH = os.path.abspath('../yolov5') 
# ---------------------

def main():
    # 1. Verify that the YOLOv5 path exists
    if not os.path.isdir(YOLOV5_PATH):
        print(f"Error: YOLOv5 path not found at '{YOLOV5_PATH}'")
        print("Please ensure the yolov5 repository is a sibling to your Adversarial_Texture project directory.")
        sys.exit(1)

    # 2. Construct the environment for the subprocess
    # We create a new PYTHONPATH that prioritizes yolov5's modules.
    new_env = os.environ.copy()
    
    # Prepend the yolov5 path so its 'utils' is found first.
    # Then add the current project path so 'yolo2', 'cfg', etc. can be found.
    project_path = os.path.abspath('.')
    new_env['PYTHONPATH'] = f"{YOLOV5_PATH}{os.pathsep}{project_path}"

    # 3. Get the path to the python interpreter currently being used
    python_executable = sys.executable

    # 4. Define the command to run our main script
    # We pass along all the command-line arguments you provide to this launcher.
    script_to_run = 'training_texture_yolov5.py'
    command = [python_executable, script_to_run] + sys.argv[1:]

    # 5. Execute the command in the new, controlled environment
    print("--- Lauching training script with isolated YOLOv5 environment ---")
    print(f"PYTHONPATH set to: {new_env['PYTHONPATH']}")
    print(f"Running command: {' '.join(command)}")
    print("-" * 60)

    try:
        subprocess.run(command, env=new_env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n--- Training script failed with exit code {e.returncode} ---")
    except KeyboardInterrupt:
        print("\n--- Training stopped by user ---")

if __name__ == '__main__':
    main()