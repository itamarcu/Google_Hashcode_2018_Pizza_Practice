from typing import List

import time
from numpy import matrix


class Rect:
    def __init__(self, r1, c1, r2, c2):
        self.r1 = r1
        self.r2 = r2
        self.c1 = c1
        self.c2 = c2


def main():
    mushroom, tomato = -1, 1
    grid: List[List[int]] = []
    R, C, L, H = 0, 0, 0, 0

    file_name = "big"

    with open(file_name + ".in") as file:
        R, C, L, H = map(int, file.readline().split(" "))
        for r in range(R):
            grid.append([mushroom if c == "M" else tomato for c in file.readline()[:C]])

    grid: matrix = matrix(grid)
    shapes: List[tuple[int, int]] = []

    lower = 2 * L
    higher = H

    for i in range(1, higher + 1):
        for j in range(1, i + 1):
            if lower <= i * j <= higher:
                if i != j:
                    shapes.append((j, i))
                shapes.append((i, j))

    def is_very_fied(r1, c1, r2, c2) -> bool:
        count_mush = 0
        count_tom = 0
        for i in range(r1, r2):
            for j in range(c1, c2):
                if grid[i, j] != mushroom and grid[i, j] != tomato:
                    return False
                count_mush += (grid[i, j] == mushroom)
                count_tom += (grid[i, j] == tomato)
        if L <= count_mush and L <= count_tom:
            return True
        return False

    def calc_key(tup):
        return tup[0] * tup[1]

    shapes.sort(key=calc_key, reverse=True)

    t1 = time.time()
    print(grid)

    rects: List[Rect] = []
    index = 10
    score = 0
    for (w, h) in shapes:
        for i in range(R - h + 1):
            for j in range(C - w + 1):
                if is_very_fied(i, j, i + h, j + w):
                    for ii in range(i, i + h):
                        for jj in range(j, j + w):
                            grid[ii, jj] = 3
                    rects.append(Rect(i, j, i + h, j + w))
                    score += w * h

    with open(file_name + ".out", mode="w") as file:
        file.write(str(len(rects)) + "\n")
        for rect in rects:
            file.write(f"{rect.r1} {rect.c1} {rect.r2 - 1} {rect.c2 - 1}\n")

    print(grid)
    t2 = time.time()
    print(f"took {t2 - t1} seconds!")
    print(f"Score is {score}")

    # for rect in rects:
    #     w = rect.c2 - rect.c1
    #     h = rect.r2 - rect.r1
    #     can_go_down = True
    #     for i in range(rect.r1, rect.r2):


main()
