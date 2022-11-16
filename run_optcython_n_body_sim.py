import sys
from optcython_n_body_sim import main

if int(len(sys.argv)) == 3:
    PROGNAME = sys.argv[0]
    N = int(sys.argv[1])
    STEPS = int(sys.argv[2])
    main(PROGNAME, N, STEPS)
else:
    print(f"Usage: {sys.argv[0]} <N> <STEPS>")