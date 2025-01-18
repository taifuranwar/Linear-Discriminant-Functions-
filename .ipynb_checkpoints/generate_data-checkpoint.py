# ...no existing code...

import csv

def generate_test_datasets():
    """Create two CSV files, each containing simple 2D data for testing."""
    dataset1 = [
        [1.0, 2.0, 0],
        [1.5, 3.0, 0],
        [2.0, 1.5, 0],
        [2.5, 2.5, 0],
        [3.0, 3.5, 0]
    ]
    with open("dataset1.csv", "w", newline="") as f1:
        writer1 = csv.writer(f1)
        for row in dataset1:
            writer1.writerow(row)

    dataset2 = [
        [4.0, 3.0, 1],
        [4.5, 2.5, 1],
        [5.0, 3.5, 1],
        [5.5, 2.0, 1],
        [6.0, 4.0, 1]
    ]
    with open("dataset2.csv", "w", newline="") as f2:
        writer2 = csv.writer(f2)
        for row in dataset2:
            writer2.writerow(row)

if __name__ == "__main__":
    generate_test_datasets()