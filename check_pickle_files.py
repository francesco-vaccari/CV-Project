import pickle
import os
import numpy as np

folder = 'transformation'
indexes = [0, 1, 2, 3, 4]

def get_all(index):
    points_file = f'points_{index}.pkl'
    old_points_file = f'old_points_{index}.pkl'
    trans_matrix_file = f'trans_matrix_{index}.pkl'
    triangle_file = f'triangles_{index}.pkl'
    old_triangle_file = f'old_triangles_{index}.pkl'

    with open(os.path.join(folder, points_file), 'rb') as f:
        points = pickle.load(f)
        print(points)

    with open(os.path.join(folder, old_points_file), 'rb') as f:
        old_points = pickle.load(f)
        print(old_points)

    with open(os.path.join(folder, trans_matrix_file), 'rb') as f:
        trans_matrix = pickle.load(f)
        print(trans_matrix)

    with open(os.path.join(folder, triangle_file), 'rb') as f:
        triangles = pickle.load(f)
        print(triangles)

    with open(os.path.join(folder, old_triangle_file), 'rb') as f:
        old_triangles = pickle.load(f)
        print(old_triangles)

for index in indexes:
    print(f'Index: {index}')
    get_all(index)
    print('--------------------')
