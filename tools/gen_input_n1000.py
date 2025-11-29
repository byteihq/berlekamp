

import sys
import random

def main():
    p = 101          
    n = 300         
    seed = 42
    if len(sys.argv) > 1:
        p = int(sys.argv[1])
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    if len(sys.argv) > 3:
        seed = int(sys.argv[3])

    random.seed(seed)

    coeffs = [random.randrange(0, p) for _ in range(n)]
    coeffs.append(1)


    out = [str(p), str(n)]
    for c in reversed(coeffs):
        out.append(str(c % p))
    print(" ".join(out))

if __name__ == "__main__":
    main()
