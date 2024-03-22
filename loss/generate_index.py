import torch


def compute_index_mat(weight, height, mini_close_number):
    index = [i for i in range(weight * height)]
    indextoxy = {}
    for i in index:
        x = i // weight
        y = i % weight
        indextoxy[i] = (x,y)

    index_matrix = torch.zeros(weight*height, mini_close_number).long()

    for i in index:
        current_x, current_y = indextoxy[i]
        dists = []
        for j in index:
            dist = (current_x - indextoxy[j][0])**2 + (current_y - indextoxy[j][1]) ** 2
            dists.append((j,dist))
        dists = sorted(dists, key=lambda x: x[1])
        min_index = [x[0] for x in dists[:mini_close_number+1]]
        index_matrix[i] = torch.tensor(min_index)[0:mini_close_number+1]
    return index_matrix





