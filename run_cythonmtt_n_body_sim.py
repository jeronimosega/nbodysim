import sys
from cythonmtt_n_body_sim import main

if int(len(sys.argv)) == 4:
    PROGNAME = sys.argv[0]
    N = int(sys.argv[1])
    STEPS = int(sys.argv[2])
    THREADS = int(sys.argv[3])
    main(PROGNAME, N, STEPS, THREADS)
else:
    print(f"Usage: {sys.argv[0]} <N> <STEPS> <THREADS>")