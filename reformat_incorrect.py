import re

def process_text_file(input_file, output_file):
    with open(input_file, 'r') as input_file:
        lines = input_file.readlines()

    data = []
    current_sample = {}

    for line in lines:
        if line.startswith("Vid") or line.startswith("Vidno"):
            current_sample['name'] = line.strip().rstrip(':')
        elif "Predicted Label" in line:
            current_sample['predicted'] = int(re.search(r'\d+', line).group())
        elif "Correct Label" in line:
            current_sample['correct'] = int(re.search(r'\d+', line).group())
            data.append(current_sample.copy())

    with open(output_file, 'w') as output_file:
        output_file.write("name\tpredicted\tcorrect\n")
        for sample in data:
            output_file.write(f"{sample['name']}\t{sample['predicted']}\t{sample['correct']}\n")

# Replace 'input.txt' and 'output.txt' with your actual file names
process_text_file('logs\IncorrectPredictionsGP.txt', 'logs\GP_errors.txt')
process_text_file('logs\IncorrectPredictionsGPM1.txt', 'logs\GPM1_errors.txt')
process_text_file('logs\IncorrectPredictionsGPM2.txt', 'logs\GPM2_errors.txt')



def subtract(file1, file2, output_file):
    # Read content from the first file
    with open(file1, 'r') as f1:
        content1 = set(line.strip() for line in f1.readlines()[1:])  # Skip header line and create a set of samples

    # Read content from the second file
    with open(file2, 'r') as f2:
        content2 = set(line.strip() for line in f2.readlines()[1:])  # Skip header line and create a set of samples

    # Find the samples that are in the first file but not in the second file
    retained_samples = content1 - content2

    # Write the retained samples to the output file
    with open(output_file, 'w') as output_file:
        output_file.write("name\tpredicted\tcorrect\n")
        for sample in retained_samples:
            output_file.write(f"{sample}\n")

def intersect(file1, file2, output_file):
    # Read content from the first file
    with open(file1, 'r') as f1:
        content1 = set(line.strip() for line in f1.readlines()[1:])  # Skip header line and create a set of samples

    # Read content from the second file
    with open(file2, 'r') as f2:
        content2 = set(line.strip() for line in f2.readlines()[1:])  # Skip header line and create a set of samples

    # Find the samples that are common between the two files
    intersecting_samples = content1.intersection(content2)

    # Write the intersecting samples to the output file
    with open(output_file, 'w') as output_file:
        output_file.write("name\tpredicted\tcorrect\n")
        for sample in intersecting_samples:
            output_file.write(f"{sample}\n")



subtract('logs\GP_errors.txt', 'logs\GPM1_errors.txt', 'logs\GPM1_corrected_errors.txt')
subtract('logs\GP_errors.txt', 'logs\GPM2_errors.txt', 'logs\GPM2_corrected_errors.txt')
subtract('logs\GPM1_errors.txt', 'logs\GP_errors.txt', 'logs\GPM1_added_errors.txt')
subtract('logs\GPM2_errors.txt', 'logs\GP_errors.txt', 'logs\GPM2_added_errors.txt')
intersect('logs\GP_errors.txt', 'logs\GPM1_errors.txt', 'logs\GPM1_consistent_errors.txt')
intersect('logs\GP_errors.txt', 'logs\GPM2_errors.txt', 'logs\GPM2_consistent_errors.txt')