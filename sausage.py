from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import torch
from math import cos, sin, pi
import time
import concurrent.futures

device = torch.get_default_device()
rotcounter90 = torch.tensor([
    [cos(pi/2), -sin(pi/2)], [sin(pi/2), cos(pi/2)]
], dtype=torch.float32).to(device)
rotclockwise90 = torch.tensor([
    [cos(-pi/2), -sin(-pi/2)], [sin(-pi/2), cos(-pi/2)]
], dtype=torch.float32).to(device)
rotcounter60 = torch.tensor([
    [cos(pi/3), -sin(pi/3)], [sin(pi/3), cos(pi/3)]
], dtype=torch.float32).to(device)

def rotate(rotmat: torch.Tensor, to_rot: torch.Tensor, focal_point: torch.Tensor) -> torch.Tensor:
    out = rotmat @ (to_rot - focal_point).T + focal_point.T
    out = out.T.to(device)
    return out

"""
Scale factor of 3, Self similar objects = 5
Hausdorff dimension of log 5/log 3 = 1.4649
"""
def quadratic_type_1(coord_list: torch.Tensor):
    start = coord_list[:, 0]
    end = coord_list[:, 1]
    dist_vec = end-start
    onethird = start + (dist_vec * 1/3)
    twothird = start + (dist_vec * 2/3)
    corner1 = rotate(rotcounter90, twothird, onethird)
    corner2 = rotate(rotclockwise90, onethird, twothird)
    out = torch.cat((
            torch.stack((start, onethird), dim=1), 
            torch.stack((onethird, corner1), dim=1), 
            torch.stack((corner1, corner2), dim=1), 
            torch.stack((corner2, twothird), dim=1), 
            torch.stack((twothird, end), dim=1)
        )).to(device)
    return out

"""
Scale factor of 4, Self similar objects = 8
Hausdorff dimension of log 8/log 4 = 1.5
"""
def sausage_curve(coord_list: torch.Tensor):
    start = coord_list[:, 0]
    end = coord_list[:, 1]
    dist_vec = end-start

    onefourth = start + (dist_vec * 1/4)
    twofourth = start + (dist_vec * 2/4)
    threefourth = start + (dist_vec * 3/4)

    top_corner1 = rotate(rotcounter90, twofourth, onefourth)
    top_corner2 = rotate(rotclockwise90, onefourth, twofourth)
    bottom_corner1 = rotate(rotclockwise90, threefourth, twofourth)
    bottom_corner2 = rotate(rotcounter90, twofourth, threefourth)
    out = torch.cat((
            torch.stack((start, onefourth), dim=1),
            torch.stack((onefourth, top_corner1), dim=1),
            torch.stack((top_corner1, top_corner2), dim=1),
            torch.stack((top_corner2, twofourth), dim=1),
            torch.stack((twofourth, bottom_corner1), dim=1),
            torch.stack((bottom_corner1, bottom_corner2), dim=1),
            torch.stack((bottom_corner2, threefourth), dim=1),
            torch.stack((threefourth, end), dim=1),
        )).to(device)
    return out

"""
Scale of 3, Self similar objects = 4
Hausdorff dimension = log 4 / log 3 = 1.2618
"""
def snowflake_curve(coord_list: torch.Tensor) -> torch.Tensor:
    start = coord_list[:, 0]
    end = coord_list[:, 1]
    dist_vec = end-start
    onethird = start + (dist_vec * 1/3)
    twothird = start + (dist_vec * 2/3)
    peak = rotate(rotcounter60, twothird, onethird)
    out = torch.cat((
            torch.stack((start, onethird), dim=1), 
            torch.stack((onethird, peak), dim=1), 
            torch.stack((peak, twothird), dim=1), 
            torch.stack((twothird, end), dim=1)
        )).to(device)
    return out

def generate(start: torch.Tensor, num_iters: int = 1, curve: Callable[[torch.Tensor], torch.Tensor] = sausage_curve) -> torch.Tensor:
    assert num_iters >= 1, "Number of iterations should be strictly positive"
    out = start
    for _ in range(num_iters):
        out = curve(out)
    return out

def generate_multiple(start: torch.Tensor, num_iters: int = 1, curve: Callable[[torch.Tensor], torch.Tensor] = sausage_curve) -> list[torch.Tensor]:
    assert num_iters >= 1, "Number of iterations should be strictly positive"
    outputs = []
    iteration = start
    for _ in range(num_iters):
        iteration = curve(iteration)
        outputs.append(torch.clone(iteration))
    return outputs

def draw_lines(lines: torch.Tensor, ax: Axes, title: str):
    for i in lines:
        ax.plot(i[:, 0], i[:, 1])
    ax.set_aspect('equal')
    ax.set_title(title)

NUM_ITERS = 4

if __name__ == "__main__":
    start = torch.tensor([
        [[0, 0], [1, 0]],
        ], dtype=torch.float32).to(device)
    begin = time.time()
    out = generate_multiple(start, NUM_ITERS, curve=sausage_curve)
    end = time.time()
    print(f"{1000*(end-begin):.3f}ms")
    fig, ax = plt.subplots(3, 2)
    fig.tight_layout()
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_ITERS) as executor:
        for it, lines in enumerate(out):
            #draw_lines(lines, ax.flat[it], f"iteration {it+1}")
            executor.submit(draw_lines, lines, ax.flat[it], f"iteration {it+1}")
    plt.show()
