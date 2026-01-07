#!/bin/bash

mkdir -p corpus
mkdir -p crashes

echo "1234" > ./corpus/init
# Use same flags as fuzzbench: https://github.com/google/fuzzbench/blob/2a2ca6ae4c5d171a52b3e20d9b7a72da306fe5b8/fuzzers/libfuzzer/fuzzer.py#L71-L92
./fuzz_binary ./corpus -fork=1 -ignore_crashes=1 -ignore_timeouts=1 -close_fd_mask=3 -artifact_prefix=./crashes/ -ignore_ooms=1 -entropic=1 -keep_seed=1 -cross_over_uniform_dist=1 -entropic_scale_per_exec_time=1 -detect_leaks=0
