import csv
from collections import defaultdict
data = defaultdict(list)
directions = {'N': 1, 'NNE': 2, 'NE': 3, 'ENE': 4, 'E': 5, 'ESE': 6, 'SE': 7, 'SSE': 8, 'S': 9,
              'SSW': 10, 'SW': 11, 'WSW': 12, 'W': 13, 'WNW': 14, 'NW': 15, 'NNW': 16, 'NA': -1, 'Yes': 1, 'No': 0}


def read_csv_file(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for ind, row in enumerate(reader):
            if ind == 0:
                continue
            for idx, num in enumerate(row):
                if num in directions:
                    row[idx] = directions[num]
                row[idx] = float(row[idx])
            data[row[-1]].append(row[:-1])

    return data
