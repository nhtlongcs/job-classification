def decompose(code):
    # major, minor, sub_minor, unit
    assert len(code) == 4, f"Invalid code: {code}"
    return code[0], code[1], code[2], code[3]


def degree(code):
    return len(code)


def distance(codeA, codeB):
    return abs(degree(codeA) - degree(codeB))


def parent(code):
    assert degree(code) > 1, f"Cannot find parent of root node: {code}"
    return code[:-1]


def LCA(u, v):
    u = "0" + u
    v = "0" + v

    def swap(u, v):
        return v, u

    if degree(u) < degree(v):
        u, v = swap(u, v)

    while degree(u) > degree(v):
        u = parent(u)

    while u != v:
        u = parent(u)
        v = parent(v)

    return degree(u) - 1


def LCA_score(codeA, codeB):
    maxDeg = 4
    return LCA(codeA, codeB) / maxDeg  # modified
    return 1 - LCA(codeA, codeB) / maxDeg  # ?


def timethis(func):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Elapsed time: {time.time() - start}")
        return result

    return wrapper


@timethis
def avg_LCA_score(codesA, codesB):
    assert len(codesA) == len(codesB), "Length mismatch"
    n = len(codesA)
    return sum(LCA_score(codesA[i], codesB[i]) for i in range(n)) / n


def random4digit():
    import random

    return "".join(str(random.randint(1, 9)) for _ in range(4))


def test():
    print(LCA_score("1111", "1111"))
    print(LCA_score("1111", "1110"))
    print(LCA_score("1111", "1210"))
    print(LCA_score("1111", "1530"))
    print("AVG")
    print(
        avg_LCA_score(
            ["1111", "1111", "1111", "1111"], ["1111", "1110", "1211", "1530"]
        )
    )
    print("gen digits")
    inp = [random4digit() for _ in range(1000000)]
    out = [random4digit() for _ in range(1000000)]
    print(avg_LCA_score(inp, out))


if __name__ == "__main__":
    # Lowest common ancestor (LCA).
    # the lowest common ancestor (LCA) (also called least common ancestor) of two nodes v and w
    # in a tree is the lowest (i.e. deepest) node that has both v and w as descendants,
    # where we define each node to be a descendant of itself.

    # The Accuracy LCA score is therefore a value between 0 and 1,
    # where higher values signify better classification performance of the method.

    test()
