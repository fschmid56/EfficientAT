NAME_TO_WIDTH = {
    'mn04_as': 0.4,
    'mn05_as': 0.5,
    'mn10_as': 1.0,
    'mn20_as': 2.0,
    'mn30_as': 3.0,
    'mn40_as': 4.0,
    'mn40_as_ext': 4.0
}


import csv

# Load label
with open('metadata/class_labels_indices.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    lines = list(reader)

labels = []
ids = []    # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)

classes_num = len(labels)
