from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x, axis = 0)
    return x

def get_covariance(dataset):
    return (np.dot(np.transpose(dataset),dataset)) / (len(dataset) - 1)

def get_eig(S, m):
    w, v = eigh(S, subset_by_index=[len(S)- m, len(S)-1])
    indices  = np.argsort(w)
    flip_indices = np.flip(indices) 

    eigenvalues = np.diag(w[flip_indices])
    eigenvectors = v[:, flip_indices]
    return eigenvalues, eigenvectors

def get_eig_prop(S, prop):

    w, v = eigh(S)
    percent = np.sum(w) * prop
    new_w, new_v = eigh(S, subset_by_value=[percent, np.inf])
    indices = np.argsort(new_w)
    flip_indices = np.flip(np.argsort(indices))

    new_eigenvalues = np.diag(new_w[flip_indices])
    new_eigenvectors = new_v[:, flip_indices]
    return new_eigenvalues, new_eigenvectors

def project_image(image, U):
    projection = np.dot(np.transpose(U), image)
    return np.dot(U, projection)

def display_image(orig, proj):
    orig = np.transpose(np.reshape(orig, (32,32)))
    proj = np.transpose(np.reshape(proj, (32,32)))

    fig,(ax1, ax2) = plt.subplots(nrows = 1, ncols = 2)
    ax1.set_title('Original')
    ax2.set_title('Projection')

    ax1_image = ax1.imshow(orig, aspect = 'equal')
    ax2_image = ax2.imshow(proj, aspect = 'equal')

    fig.colorbar(ax1_image, ax = ax1)
    fig.colorbar(ax2_image, ax = ax2)
    plt.show()
