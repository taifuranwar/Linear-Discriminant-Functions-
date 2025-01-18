import numpy as np
import matplotlib.pyplot as plt
import csv
from ipywidgets import interact, FloatSlider, Checkbox
import os
from tkinter import Tk, filedialog

def load_dataset(csv_file):
    """Load the dataset from a CSV file."""
    if not (os.path.exists("dataset1.csv") and os.path.exists("dataset2.csv")):
        root = Tk()
        root.withdraw()
        selected_files = filedialog.askopenfilenames(
            title="Datasets not found. Please select two CSV files.",
            filetypes=[("CSV files", "*.csv")]
        )
        if len(selected_files) != 2:
            raise ValueError("You must select exactly two CSV files.")
        root.destroy()

    data = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            x, y, label = row
            data.append((float(x), float(y), int(label)))
    return np.array(data)

def linear_discriminant(x, w):
    """Compute the linear discriminant."""
    # x is a 1D array: [x1, x2], w is [w0, w1, w2]
    # Discriminant: g(x) = w0 + w1*x1 + w2*x2
    return np.dot(w, np.array([1, x[0], x[1]]))

def sigmoid(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-z))

def generalized_discriminant(x, w):
    """Apply a sigmoid to the linear discriminant."""
    return sigmoid(linear_discriminant(x, w))

def plot_datasets(dataset1, dataset2):
    """Plot two datasets on a 2D plane."""
    plt.scatter(dataset1[:,0], dataset1[:,1], c='blue', label='Class 1')
    plt.scatter(dataset2[:,0], dataset2[:,1], c='red', label='Class 2')
    plt.legend()

def plot_discriminant_boundary(w, dataset1, dataset2, use_sigmoid=False):
    """Plot the discriminant boundary and isoclines (contours)."""
    plt.figure(figsize=(6,6))
    plot_datasets(dataset1, dataset2)
    
    # Generate a grid
    x_min, x_max = min(dataset1[:,0].min(), dataset2[:,0].min()) - 1, max(dataset1[:,0].max(), dataset2[:,0].max()) + 1
    y_min, y_max = min(dataset1[:,1].min(), dataset2[:,1].min()) - 1, max(dataset1[:,1].max(), dataset2[:,1].max()) + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    zz = np.zeros(xx.shape)
    
    # Fill the grid with discriminant values
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = np.array([xx[i,j], yy[i,j]])
            if use_sigmoid:
                zz[i,j] = generalized_discriminant(point, w)
            else:
                zz[i,j] = linear_discriminant(point, w)
    
    # Plot the contour for discriminant = 0 or for sigmoid = 0.5
    level = 0.5 if use_sigmoid else 0.0
    contour = plt.contour(xx, yy, zz, levels=[level], colors='green')
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Plot more isoclines for visualization
    if use_sigmoid:
        # e.g. 0.2, 0.8, etc.
        levels = [0.2, 0.8]
    else:
        # e.g. -1, 1, etc.
        levels = [-1, 1]
    contour_isoclines = plt.contour(xx, yy, zz, levels=levels, linestyles='--', colors='gray')
    plt.clabel(contour_isoclines, inline=True, fontsize=8)
    
    plt.title("Discriminant Visualization (Sigmoid: {})".format(use_sigmoid))
    plt.show()

def interactive_visualization(csv1, csv2):
    """Create an interactive widget to visualize the discriminant."""
    dataset1 = load_dataset(csv1)
    dataset1_xy = dataset1[dataset1[:,2]==0][:,0:2]
    dataset2 = load_dataset(csv2)
    dataset2_xy = dataset2[dataset2[:,2]==1][:,0:2]
    
    def update(w0=0.0, w1=1.0, w2=1.0, use_sigmoid=False):
        w = np.array([w0, w1, w2])
        plot_discriminant_boundary(w, dataset1_xy, dataset2_xy, use_sigmoid)
    
    interact(
        update,
        w0=FloatSlider(min=-5, max=5, step=0.1, value=0),
        w1=FloatSlider(min=-5, max=5, step=0.1, value=1),
        w2=FloatSlider(min=-5, max=5, step=0.1, value=1),
        use_sigmoid=Checkbox(value=False, description='Use Sigmoid')
    )

def main():
    interactive_visualization("dataset1.csv", "dataset2.csv")

if __name__ == "__main__":
    main()