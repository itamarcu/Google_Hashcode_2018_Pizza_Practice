import random
from typing import List, Tuple
import numpy as np
import time


def char_to_bit(c):
    if c == "M":
        return mushroom
    if c == "T":
        return tomato
    print("ERROR! this char was encountered: " + c)
    return None


class Rect:
    
    def __init__(self, r1, c1, r2, c2):
        self.r1 = r1
        """First row of this rectangle.
        
        INCLUSIVE.
        ----
        !"""
        self.r2 = r2
        """Last row of this rectangle.
        
        INCLUSIVE.
        ----
        !"""
        self.c1 = c1
        """First column of this rectangle.
        
        EXCLUSIVE.
        ----
        !"""
        self.c2 = c2
        """Last column of this rectangle.
        
        EXCLUSIVE.
        ----
        !"""


def check_eligible(grid_bits, grid_taken, r1, c1, r2, c2, L) -> bool:
    count_mush = 0
    count_tom = 0
    for i in range(r1, r2):
        for j in range(c1, c2):
            if grid_taken[i, j]:
                return False
            if grid_bits[i, j] == mushroom:
                count_mush += 1
            else:
                count_tom += 1
    if L <= count_mush and L <= count_tom:
        return True
    return False


def key_area(tup):
    return tup[0] * tup[1]


# This function greatly affects the score!!!"""
calc_key = key_area


def solve() -> List[Rect]:
    with open(file_name + ".in") as file:
        R, C, L, H = map(int, file.readline().split(" "))
        grid_bits = np.zeros(shape=(R, C), dtype=np.bool)
        for r in range(R):
            line = file.readline()[:C]
            for c in range(len(line)):
                grid_bits[r, c] = char_to_bit(line[c])
    
    grid_taken = np.zeros(grid_bits.shape, dtype=np.bool)
    shapes: List[Tuple[int, int]] = []
    lower_bound = 2 * L
    upper_bound = H
    
    if print_stuff:
        print(f"Solving {file_name} with R={R}, C={C}, L={L}, H={H}...")
        print("GRID BITS")
        print(grid_bits.astype(np.int))
    
    for i in range(1, upper_bound + 1):
        for j in range(1, i + 1):
            if lower_bound <= i * j <= upper_bound:
                if i != j:
                    shapes.append((j, i))
                shapes.append((i, j))
    
    if greedy:
        shapes.sort(key=calc_key, reverse=True)
    else:
        shapes.sort(key=calc_key, reverse=False)
    rectangles_in_cover: List[Rect] = []
    score = 0
    
    t1 = time.time()
    for (w, h) in shapes:
        for i in range(R - h + 1):
            for j in range(C - w + 1):
                if check_eligible(grid_bits, grid_taken, i, j, i + h, j + w, L):
                    for ii in range(i, i + h):
                        for jj in range(j, j + w):
                            grid_taken[ii, jj] = True
                    rectangles_in_cover.append(Rect(i, j, i + h, j + w))
                    score += w * h
    t2 = time.time()
    
    if print_stuff:
        print("GRID TAKEN")
        print(grid_taken.astype(np.int))
        print(f"Time took (s):    {t2 - t1}")
        print(f"Score:    {score}/{R*C}")
    
    if greedy:
        return rectangles_in_cover
    
    if print_stuff:
        print()
        print("Let the expansion begin!")
    
    t1 = time.time()
    for rect in rectangles_in_cover:
        w = rect.c2 - rect.c1
        h = rect.r2 - rect.r1
        area = w * h
        leftover_possible = upper_bound - area
        if leftover_possible <= 0:  # this shouldn't ever be negative but I
            # think this check is faster
            continue
        while True:
            # DOWN
            if leftover_possible >= w \
                    and rect.r2 + 1 <= R \
                    and not grid_taken[rect.r2, rect.c1:rect.c2].any():
                grid_taken[rect.r2, rect.c1:rect.c2] = True
                h += 1
                score += w
                area += w
                leftover_possible -= w
                rect.r2 += 1
            # RIGHT
            elif leftover_possible >= h \
                    and rect.c2 + 1 <= C \
                    and not grid_taken[rect.r1: rect.r2, rect.c2].any():
                grid_taken[rect.r1: rect.r2, rect.c2] = True
                w += 1
                score += h
                area += h
                leftover_possible -= h
                rect.c2 += 1
            # UP
            elif leftover_possible >= w \
                    and rect.r1 - 1 >= 0 \
                    and not grid_taken[rect.r1 - 1, rect.c1:rect.c2].any():
                grid_taken[rect.r1 - 1, rect.c1:rect.c2] = True
                h += 1
                score += w
                area += w
                leftover_possible -= w
                rect.r1 -= 1
            # LEFT
            elif leftover_possible >= h \
                    and rect.c1 - 1 >= 0 \
                    and not grid_taken[rect.r1: rect.r2, rect.c1 - 1].any():
                grid_taken[rect.r1: rect.r2, rect.c1 - 1] = True
                w += 1
                score += h
                area += h
                leftover_possible -= h
                rect.c1 -= 1
            else:
                break
    t2 = time.time()
    
    if print_stuff:
        print("GRID TAKEN")
        print(grid_taken.astype(np.int))
        print(f"Time took (s):    {t2 - t1}")
        print(f"Score:    {score}/{R*C}")
    
    return rectangles_in_cover


def calc_score(rectangles_in_cover: List[Rect]) -> int:
    return sum([(rect.r2 - rect.r1) * (rect.c2 - rect.c1) for rect in
                rectangles_in_cover])


def write_solution_to_file(rectangles_in_cover: List[Rect]):
    score = calc_score(rectangles_in_cover)
    write_name = f"{file_name}_{score}.out"
    with open(write_name, mode="w") as file:
        file.write(str(len(rectangles_in_cover)) + "\n")
        for rect in rectangles_in_cover:
            file.write(f"{rect.r1} {rect.c1} {rect.r2 - 1} {rect.c2 - 1}\n")
    print(f"Written score {score} to file {write_name}!")


keys_given = dict()


def key_random_remember(tup):
    random_key = random.randint(0, 1000000)
    keys_given[tup] = random_key
    return random_key


#
#
#
# Variables us programmers can mess with
mushroom, tomato = True, False
np.set_printoptions(edgeitems=5)
file_name = ["example", "small", "medium", "big"][3]
greedy = False
print_stuff = False
random_attempts = True
#
#
#

if random_attempts:
    calc_key = key_random_remember
    best_solution = None
    best_score = -1
    for iteration in range(67):
        keys_given.clear()
        solution = solve()
        score = calc_score(solution)
        if best_score < score:
            best_score = score
            best_solution = solution
            print(f"\nIteration {iteration}: new best is {best_score}")
            print("Random key map:")
            shapes = list(keys_given.keys())
            shapes.sort(key=lambda shape: keys_given[shape])
            print(shapes)
            write_solution_to_file(best_solution)
        else:
            print(f"{iteration}: {score}...")
    print("\nFINISHED")
    write_solution_to_file(best_solution)
else:
    solution = solve()
    write_solution_to_file(solution)
