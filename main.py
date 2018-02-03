import random
import string
import time
from typing import List, Tuple, Callable, TypeVar, Dict

import numpy as np


class Rect:
    """_o variables are original, other variables are extended"""
    
    def __init__(self, r1, c1, r2, c2):
        self.r1_o = r1
        """First row of this rectangle.

        INCLUSIVE.
        ----
        !"""
        self.r2_o = r2
        """Last row of this rectangle.

        INCLUSIVE.
        ----
        !"""
        self.c1_o = c1
        """First column of this rectangle.

        EXCLUSIVE.
        ----
        !"""
        self.c2_o = c2
        """Last column of this rectangle.

        EXCLUSIVE.
        ----
        !"""
        self.r1 = self.r1_o
        self.c1 = self.c1_o
        self.r2 = self.r2_o
        self.c2 = self.c2_o
    
    def reset_extensions(self):
        self.r1 = self.r1_o
        self.c1 = self.c1_o
        self.r2 = self.r2_o
        self.c2 = self.c2_o
    
    def area_o(self):
        return (self.r2_o - self.r1_o) * (self.c2_o - self.c1_o)
    
    def area_x(self):
        return (self.r2 - self.r1) * (self.c2 - self.c1)


class BaseData(object):
    def __init__(self, file_name):
        self.L: int = 0
        self.H: int = 0
        self.C: int = 0
        self.R: int = 0
        self.file_name: str = file_name
        self.grid_bits: np.multiarray = np.zeros(shape=(0, 0), dtype=np.bool)
        self.possible_rects: List[Rect] = []
        self.shapes: List[Tuple[int, int]] = []


RectKeyFunction = TypeVar("Rectangle Key Function", bound=Callable[[Rect], int])
Permutation = TypeVar("Permutation", bound=List[int])


def char_to_bit(c: str):
    if c == "M":
        return mushroom
    if c == "T":
        return tomato
    print("ERROR! this char was encountered: " + c)
    return None


def setup(file_name: str) -> BaseData:
    base_data: BaseData = BaseData(file_name)
    with open(base_data.file_name + ".in") as file:
        base_data.R, base_data.C, \
        base_data.L, base_data.H = map(int, file.readline().split(" "))
        base_data.grid_bits = np.zeros(shape=(base_data.R, base_data.C),
                                       dtype=np.bool)
        for r in range(base_data.R):
            line = file.readline()[:base_data.C]
            for c in range(base_data.C):
                base_data.grid_bits[r, c] = char_to_bit(line[c])
    
    if print_stuff:
        print(f"Solving {base_data.file_name}"
              f" with R={base_data.R}, C={base_data.C},"
              f" L={base_data.L}, H={base_data.H}...")
        print("GRID BITS")
        print(base_data.grid_bits.astype(np.int))
    
    base_data.shapes = []
    for i in range(1, base_data.H + 1):
        for j in range(1, i + 1):
            if 2 * base_data.L <= i * j <= base_data.H:
                if i != j:
                    base_data.shapes.append((j, i))
                base_data.shapes.append((i, j))
    
    sliding_window_validation(base_data)
    
    return base_data


def sliding_window_validation(base_data: BaseData):
    print("Setting up list (this takes a long one-time up-front cost)"
          " ...", end="", flush=True)
    base_data.possible_rects = []
    for (w, h) in base_data.shapes:
        print(".", end="", flush=True)
        area = w * h
        for i in range(base_data.R - h + 1):
            # Fully calculate for first window in row
            # j = 0 for this first one
            bit_count = base_data.grid_bits[i:i + h, 0:0 + w] \
                .astype(np.int).sum()
            if base_data.L <= bit_count and base_data.L <= area - bit_count:
                base_data.possible_rects.append(Rect(i, 0, i + h, 0 + w))
            for j in range(1, base_data.C - w + 1):
                # subtract bits one column left of leftmost column
                bit_count -= base_data.grid_bits[i:i + h, j - 1] \
                    .astype(np.int).sum()
                # add bits of rightmost column
                bit_count += base_data.grid_bits[i:i + h, j + w - 1] \
                    .astype(np.int).sum()
                if base_data.L <= bit_count and base_data.L <= area - bit_count:
                    base_data.possible_rects.append(Rect(i, j, i + h, j + w))
    print("done.")


def solve(base_data: BaseData):
    for rect in base_data.possible_rects:
        rect.reset_extensions()
    t1 = time.time()
    grid_takens = np.zeros(base_data.grid_bits.shape, dtype=np.bool)
    rectangles_in_cover: List[Rect] = []
    for rect in base_data.possible_rects:
        if check_free(grid_takens, rect):
            grid_takens[rect.r1:rect.r2, rect.c1:rect.c2] = True
            rectangles_in_cover.append(rect)
    t2 = time.time()
    
    if print_stuff:
        print("GRID TAKEN")
        print(grid_takens.astype(np.int))
        print(f"Time took (s):    {t2 - t1}")
        print(f"Score:    {calc_score(rectangles_in_cover)}"
              f"/{base_data.R*base_data.C}")
    
    stretch(base_data, rectangles_in_cover, grid_takens)
    
    return rectangles_in_cover


def check_free(grid_takens: np.ndarray, rect: Rect) -> bool:
    return not grid_takens[rect.r1:rect.r2, rect.c1:rect.c2].any()


def calc_score(rectangles_in_cover: List[Rect]) -> int:
    return sum([rect.area_x() for rect in
                rectangles_in_cover])


def stretch(base_data: BaseData, rectangles_in_cover: List[Rect],
            grid_takens: np.ndarray):
    if print_stuff:
        print()
        print("Let the expansion begin!")
    t1 = time.time()
    for rect in rectangles_in_cover:
        w = rect.c2 - rect.c1
        h = rect.r2 - rect.r1
        area = w * h
        leftover_possible = base_data.H - area
        if leftover_possible <= 0:  # this shouldn't ever be negative but I
            # think this check is faster
            continue
        while True:
            # DOWN
            if leftover_possible >= w \
                    and rect.r2 + 1 <= base_data.R \
                    and not grid_takens[rect.r2, rect.c1:rect.c2].any():
                grid_takens[rect.r2, rect.c1:rect.c2] = True
                h += 1
                area += w
                leftover_possible -= w
                rect.r2 += 1
            # RIGHT
            elif leftover_possible >= h \
                    and rect.c2 + 1 <= base_data.C \
                    and not grid_takens[rect.r1: rect.r2, rect.c2].any():
                grid_takens[rect.r1: rect.r2, rect.c2] = True
                w += 1
                area += h
                leftover_possible -= h
                rect.c2 += 1
            # UP
            elif leftover_possible >= w \
                    and rect.r1 - 1 >= 0 \
                    and not grid_takens[rect.r1 - 1, rect.c1:rect.c2].any():
                grid_takens[rect.r1 - 1, rect.c1:rect.c2] = True
                h += 1
                area += w
                leftover_possible -= w
                rect.r1 -= 1
            # LEFT
            elif leftover_possible >= h \
                    and rect.c1 - 1 >= 0 \
                    and not grid_takens[rect.r1: rect.r2, rect.c1 - 1].any():
                grid_takens[rect.r1: rect.r2, rect.c1 - 1] = True
                w += 1
                area += h
                leftover_possible -= h
                rect.c1 -= 1
            else:
                break
    t2 = time.time()
    
    if print_stuff:
        print("GRID TAKEN")
        print(grid_takens.astype(np.int))
        print(f"Time took (s):    {t2 - t1}")
        print(f"Score:    {calc_score(rectangles_in_cover)}"
              f"/{base_data.R*base_data.C}")


def write_solution_to_file(base_data: BaseData,
                           rectangles_in_cover: List[Rect]):
    score = calc_score(rectangles_in_cover)
    write_name = f"{base_data.file_name}_{score}.out"
    with open(write_name, mode="w") as file:
        file.write(str(len(rectangles_in_cover)) + "\n")
        for rect in rectangles_in_cover:
            file.write(f"{rect.r1} {rect.c1} {rect.r2 - 1} {rect.c2 - 1}\n")
    print(f"Written score {score} to file {write_name}!")


def key_greedy(rect: Rect):
    return rect.area_o()


def key_r2c2(rect: Rect):
    return rect.r2, rect.c2


def key_ungreedy_topleft(rect: Rect):
    return - rect.area_o(), rect.r1, rect.c1


def key_ungreedy_r2c2(rect: Rect):
    return - rect.area_o(), rect.r2, rect.c2


def key_ungreedy_bottomright(rect: Rect):
    return - rect.area_o(), -rect.r1, -rect.c1


def solve_once(base_data: BaseData, key: RectKeyFunction):
    base_data.possible_rects.sort(key=key)
    solution = solve(base_data)
    write_solution_to_file(base_data, solution)


def solve_random_attempts(base_data: BaseData, rng: random.Random):
    random_weights = []
    
    def key_random_weights(rect: Rect):
        if len(random_weights) == 0:
            random_weights.extend(
                    [rng.uniform(-1, 1), rng.uniform(-1, 1),
                     rng.uniform(-1, 1)])
        return rect.r1 * random_weights[0] + rect.c1 * random_weights[1] \
               + rect.area_x() * random_weights[2]
    
    best_score = -1
    for iteration in range(9999):
        seed = "".join(rng.choice(string.ascii_letters) for _ in range(64))
        rng = random.Random(seed)
        random_weights = []
        base_data.possible_rects.sort(key=key_random_weights)
        solution = solve(base_data)
        score = calc_score(solution)
        if best_score < score:
            best_score = score
            print(f"\nIteration {iteration}: new best is {best_score}."
                  f" Random seed was {seed}")
            write_solution_to_file(base_data, solution)
        else:
            print(f"{iteration}: {score}...")
    print("\nFINISHED")


def genetic_algorithm_1(base_data: BaseData, rng: random.Random):
    n = len(base_data.possible_rects)
    current_best_selection_table: List[int] = [0 for _ in range(n)]
    best_score = calc_score(solve(base_data))
    for i in range(1, 1234):
        random_selection_table = current_best_selection_table.copy()
        mutate_table(random_selection_table, rng, chance=0.1)
        sort_rects_by_table(base_data.possible_rects, random_selection_table)
        solution = solve(base_data)
        score = calc_score(solution)
        if best_score < score:
            best_score = score
            current_best_selection_table = random_selection_table
            print(f"iteration {i}: {score} !!!"
                  f"        {[rect.r1 for rect in solution[:10]]}")
        else:
            print(f"iteration {i}: {score}"
                  f"        {[rect.r1 for rect in solution[:10]]}")
        undo_sort_by_table(base_data.possible_rects, random_selection_table)


def sort_rects_by_table(list_rects: List[Rect], selection_table: List[int]):
    for i in range(len(list_rects)):
        j = i + selection_table[i]
        list_rects[i], list_rects[j] = list_rects[j], list_rects[i]


def undo_sort_by_table(list_rects: List[Rect], selection_table: List[int]):
    for i in range(len(list_rects) - 1, 0 - 1, -1):
        j = i + selection_table[i]
        list_rects[i], list_rects[j] = list_rects[j], list_rects[i]


def mutate_table(selection_table: List[int], rng: random.Random, chance: float):
    n = len(selection_table)
    for i in range(n - 1):  # last one is always 0
        if rng.random() <= chance:
            nudge = rng.choice([-1, +1]) * n // 3
            selection_table[i] = (selection_table[i] + nudge) % (n - i - 1)


def genetic_algorithm_2(base_data: BaseData, rng: random.Random):
    n = len(base_data.possible_rects)
    current_best_permutation: Permutation = list(range(n))
    original_order = base_data.possible_rects
    best_score = calc_score(solve(base_data))
    for i in range(1, 1234):
        random_permutation = current_best_permutation.copy()
        mutate_permutation(random_permutation, rng, times=97)
        base_data.possible_rects = sort_rects_by_permutation(
                base_data.possible_rects, random_permutation)
        solution = solve(base_data)
        score = calc_score(solution)
        if best_score < score:
            best_score = score
            current_best_permutation = random_permutation
            print(f"iteration {i:04}: {score} !!!"
                  f"        {[rect.c2 for rect in solution[:10]]}")
        else:
            print(f"iteration {i:04}: {score}    "
                  f"        {[rect.c2 for rect in solution[:10]]}")
        base_data.possible_rects = original_order


def mutate_permutation(permutation: Permutation, rng: random.Random,
                       times: int):
    n = len(permutation)
    for _ in range(times):
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        permutation[i], permutation[j] = permutation[j], permutation[i]


def sort_rects_by_permutation(list_rects: List[Rect], permutation: Permutation) \
        -> List[Rect]:
    return [list_rects[x] for x in permutation]
    # O(n) memory, but this can be optimized to O(1)


def genetic_algorithm_3(base_data: BaseData, rng: random.Random):
    n = len(base_data.possible_rects)
    population_size = 4  # should be even
    adults = [list(range(n)) for _ in range(population_size)]
    for adult in adults:
        random.shuffle(adult)
    children = []
    original_order = base_data.possible_rects
    best_score = -1
    for iteration in range(1, 1234):
        children.clear()
        rng.shuffle(adults)
        for j in range(0, population_size, 2):
            children.extend(
                    permutation_copulation(adults[j], adults[j + 1], rng))
        adults.extend(children)
        scores: Dict[Permutation, int] = {}
        for index, permutation in enumerate(adults):
            base_data.possible_rects = sort_rects_by_permutation(
                    base_data.possible_rects, permutation)
            solution = solve(base_data)
            scores[index] = calc_score(solution)
        adult_indexes = list(range(len(adults)))
        adult_indexes.sort(key=lambda idx: scores[idx], reverse=True)
        adult_indexes = adult_indexes[:population_size]  # decimate
        adults = [adults[idx] for idx in adult_indexes]
        best_score_in_iteration = max(scores.values())
        if best_score < best_score_in_iteration:
            best_score = best_score_in_iteration
            print(f"iteration {iteration:04}: {best_score_in_iteration} !!!"
                  f"        ")
        else:
            print(f"iteration {iteration:04}: {best_score_in_iteration}    "
                  f"        ")
        base_data.possible_rects = original_order


def permutation_copulation(a1: Permutation, a2: Permutation,
                           rng: random.Random) -> List[Permutation]:
    """One child will be similar to parent 1, and the other to parent 2."""
    n = len(a1)
    start = rng.randint(0, n - 1)
    end = rng.randint(0, n)
    if end < start:
        start, end = end, start
    child_1 = list()
    child_1.extend(a1[start:end])
    child_1_set = set(child_1)
    child_2 = a2[start:end]
    child_2_set = set(child_2)
    for i in range(n):
        current_index = (end + i) % n
        item_1 = a1[current_index]
        item_2 = a2[current_index]
        if item_2 not in child_1_set:
            child_1_set.add(item_2)
            child_1.append(item_2)
        if item_1 not in child_2_set:
            child_2_set.add(item_1)
            child_2.append(item_1)
    
    # rotate
    child_1 = child_1[start:] + child_1[:start]
    child_2 = child_2[start:] + child_2[:start]
    return [child_1, child_2]


def main():
    rng = random.Random(time.time())
    base_data = setup("big")
    solve_random_attempts(base_data, rng)
    # solve_once(base_data, key_ungreedy_r2c2)
    # genetic_algorithm_2(base_data, rng)


#
#
#
# Variables us programmers can mess with
mushroom, tomato = True, False
np.set_printoptions(edgeitems=5)
print_stuff = False
random_attempts = True
#
#
#

if __name__ == '__main__':
    main()
