#!/bin/bash
srun --nodes 4 --tasks-per-node 18 --account ACCOUNT_ID_HERE --time 00:01:00 ./matmul -a a.in -b b.in -g 0 -t 3D -l 2