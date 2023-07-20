import os
import subprocess
from tqdm import tqdm
import argparse

path1 = "/home/azureuser/interpret_bert/text_probing"
path2 = "/mnt/bert_features"

tasks = os.listdir(''.join((path1, "/task_datasets")))

def run_command(task_name, layer, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)  # Use the specified GPU
    command_output = subprocess.check_output(["python",
                                            ''.join((path1, "/classifier.py")),
                                            "--labels_file",
                                            ''.join((path1, "/task_datasets/", task_name, ".txt")),
                                            "--feats_file",
                                            ''.join((path2, "/", task_name, "_rep.json")),
                                            "--layer", str(layer),
                                            "--gpu", str(gpu)],
                                            text=True,
                                            env=env)

    # Split the command_output by new lines
    output_lines = command_output.strip().split("\n")

    # Get the last line (assuming it contains the dev score and test score)
    last_line = output_lines[-1]

    # Split the last line by semicolon to extract dev score and test score
    last_line_parts = last_line.split(";")

    # Extract the numeric values for dev score and test score
    dev_score = float(last_line_parts[1].split("=")[1].strip())
    test_score = float(last_line_parts[2].split("=")[1].strip())

    return dev_score, test_score

def main():
    parser = argparse.ArgumentParser(description="Process tasks and layers for a specific task on a GPU.")
    parser.add_argument("task", choices=tasks, help="The specific task to process.")
    parser.add_argument("--gpu", type=int, default=0, choices=[0, 1, 2, 3], help="The GPU number to use (0, 1, 2, or 3).")

    args = parser.parse_args()

    task = args.task
    gpu = args.gpu

    validation_output_file = f"{task}_validation_output.txt"
    test_output_file = f"{task}_test_output.txt"

    with open(validation_output_file, "w") as validation_file, open(test_output_file, "w") as test_file:
        # Write task name to files
        validation_file.write(f"{task}\n")
        test_file.write(f"{task}\n")

        for layer in tqdm(range(12), desc=f"Processing {task}"):
            dev_score, test_score = run_command(task.split('.')[0], layer, gpu)

            # Write dev score and test score for the current layer
            validation_file.write(f"{dev_score}\n")
            test_file.write(f"{test_score}\n")

if __name__ == "__main__":
    main()
