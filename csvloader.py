import csv
import torch


def load(filename: str):
    rows = []
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if reader.line_num == 1:
                continue
            rows.append(list(map(lambda x: float(x), row)))

    return torch.tensor(rows)
