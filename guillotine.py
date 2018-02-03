import numpy as np


def solve():
    with open(file_name + ".in") as file:
        R, C, L, H = map(int, file.readline().split(" "))
        grid_bits = np.zeros(shape=(R, C), dtype=np.bool)
        for r in range(R):
            line = file.readline()[:C]
            for c in range(len(line)):
                grid_bits[r, c] = mushroom if line[c] == 'M' else tomato
    print(grid_bits.astype(np.int))
    
    best_cut_values = {}
    best_cuts = {}  # negative means it's a vertical cut, and not horizontal
    order = []
    for r in range(R):
        for c in range(C):
            for h in range(1, R - r + 1):
                for w in range(1, C - c + 1):
                    order.append((r, c, h, w))
    print(len(order))
    order.sort(key=lambda tup: tup[2] * tup[3])
    print("got sorted")
    for r, c, h, w in order:  # increasing order of size
        print(r, c, w, h)
        best_cut_values[r, c, h, w] = 0
        best_cuts[r, c, h, w] = 0  # zero: no cut
        c0, c1 = 0, 0
        for x in range(h):
            for y in range(w):
                if grid_bits[r + x, c + y]:
                    c1 += 1
                else:
                    c0 += 1
        if c0 < L or c1 < L:  # base case: impossible to cover
            continue
        if h * w <= H:  # base case: can be fully covered with a single slice
            best_cut_values[r, c, h, w] = w * h
            continue
        for cut in range(1, h):
            # val is a possible score, the score for this region for this cut
            val = best_cut_values[r, c, cut, w] + best_cut_values[
                r + cut, c, h - cut, w]
            if val > best_cut_values[r, c, h, w]:
                best_cut_values[r, c, h, w] = val
                best_cuts[r, c, h, w] = cut  # horizontal cut
        for cut in range(1, w):
            val = best_cut_values[r, c, h, cut] + best_cut_values[
                r, c + cut, h, w - cut]
            if val > best_cut_values[r, c, h, w]:
                best_cut_values[r, c, h, w] = val
                best_cuts[r, c, h, w] = -cut  # vertical cut
    
    sol = []  # retrieve solution
    
    stack = [(0, 0, R, C)]
    while stack:
        ssss = stack.pop()
        r, c, h, w = ssss
        cut = best_cuts[r, c, h, w]
        if cut == 0:  # this rectangle is valid
            if best_cut_values[r, c, h, w]:
                sol.append((r, c, h, w))
        elif cut > 0:
            stack.append((r, c, cut, w))
            stack.append((r + cut, c, h - cut, w))
        else:
            stack.append((r, c, h, -cut))
            stack.append((r, c - cut, h, w + cut))
    
    print(f"SCORE: {sum([h*w for (r, c, h, w) in sol])}")
    # for (r, c, h, w) in sol:


#
#
mushroom, tomato = True, False
np.set_printoptions(edgeitems=5)
file_name = ["example", "small", "medium", "big"][2]

solve()
