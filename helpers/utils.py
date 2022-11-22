def NAME_TO_WIDTH(name):
    map = {
        'mn04': 0.4,
        'mn05': 0.5,
        'mn10': 1.0,
        'mn20': 2.0,
        'mn30': 3.0,
        'mn40': 4.0
    }
    return map[name[:4]]


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
