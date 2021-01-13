#!/bin/bash

rsync --exclude '*data*' -avz -e ssh uqfcogno@wiener.hpc.net.uq.edu.au:/scratch/cai/francesco/PialNet/francesco/checkpoints/* francesco/checkpoints