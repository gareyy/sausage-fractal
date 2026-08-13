import matplotlib.pyplot as plt
import numpy as np
import torch
from math import cos, sin, pi
from tqdm import tqdm
import time
# TODO: add zooming in
# TODO: add type 2 quadratic
device = torch.get_default_device()

absdist = torch.nn.PairwiseDistance(p=1)

rotcounter90 = torch.tensor([
    [cos(pi/2), -sin(pi/2)], [sin(pi/2), cos(pi/2)]
], dtype=torch.float32).to(device)
rotclockwise90 = torch.tensor([
    [cos(-pi/2), -sin(-pi/2)], [sin(-pi/2), cos(-pi/2)]
], dtype=torch.float32).to(device)

def quadratic_type_1(coord_list: torch.Tensor):
    start = coord_list[:, 0]
    end = coord_list[:, 1]
    dist_vec = end-start
    onethird = start + (dist_vec * 1/3)
    twothird = start + (dist_vec * 2/3)
    corner1 = rotcounter90 @ (twothird - onethird).T + onethird.T
    corner1 = corner1.T
    corner2 = rotclockwise90 @ (onethird - twothird).T + twothird.T
    corner2 = corner2.T

    out = (
            torch.stack((start, onethird), dim=1), 
            torch.stack((onethird, corner1), dim=1), 
            torch.stack((corner1, corner2), dim=1), 
            torch.stack((corner2, twothird), dim=1), 
            torch.stack((twothird, end), dim=1)
        )
    out = torch.cat(out)
    return out

if __name__ == "__main__":
    start = torch.tensor([
        [[0, 0], [1, 0]],
        ], dtype=torch.float32).to(device)
    begin = time.time()
    out = start
    for i in range(5):
        out = quadratic_type_1(out)
    end = time.time()
    print(f"{1000*(end-begin):.3f}ms")
    for p in out:
        plt.plot(p[:, 0], p[:, 1])
    plt.gca().set_aspect('equal')
    plt.show()
