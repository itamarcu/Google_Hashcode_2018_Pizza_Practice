from typing import List
from numpy import matrix

import math


class Rect:
    def __init__(self, r1, c1, r2, c2):
        self.r1 = r1
        self.r2 = r2
        self.c1 = c1
        self.c2 = c2


def main():
    grid: List[List[int]] = []
    R, C, L, H = 0, 0, 0, 0
    file_name = "medium"

    with open(file_name + ".in") as file:
        R, C, L, H = map(int, file.readline().split(" "))
        for r in range(R):
            grid.append([1 if c == "M" else 0 for c in file.readline()])

    grid: matrix = matrix(grid)

    valid_rects: List[Rect] = []
    overlaps: List[List[List[int]]] = [[[] for _ in range(C)] for _ in range(R)]
    shapes: List[tuple[int, int]] = []

    lower = 2 * L
    higher = H
    delta = higher - lower

    for i in range(1, higher + 1):
        for j in range(1, i + 1):
            if lower <= i * j <= higher:
                shapes.append((i, j))
                if i != j:
                    shapes.append((j, i))

    def is_very_fied(r1, c1, r2, c2) -> bool:
        seen_one = False
        seen_zero = False
        for i in range(r1, r2):
            for j in range(c1, c2):
                seen_one = seen_one or (grid[i, j] == 1)
                seen_zero = seen_zero or (grid[i, j] == 0)
                if seen_one and seen_zero:
                    return True
        return False

    for (w, h) in shapes:
        for i in range(R - h + 1):
            for j in range(C - w + 1):
                if is_very_fied(i, j, i + h, j + w):
                    rect = Rect(i, j, i + h, j + w)
                    valid_rects.append(rect)
                    for ii in range(i, i + h):
                        for jj in range(j, j + w):
                            overlaps[ii][jj].append(len(valid_rects) - 1)  # that's the index

    overlap_count_grid = matrix([[len(overlaps[i][j]) for j in range(C)] for i in range(R)])
    print(overlap_count_grid)
    print(grid)
    print(len(valid_rects))



if __name__ == '__main__':
    main()
