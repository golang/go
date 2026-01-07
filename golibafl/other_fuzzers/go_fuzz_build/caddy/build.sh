#!/bin/bash

go-118-fuzz-build -tags gofuzz -o fuzz_archive_file.a -func FuzzMe .

#clang -o fuzz_binary fuzz_archive_file.a -fsanitize=fuzzer
# link with libfuzzer runtime
#clang -o fuzz_binary fuzz_archive_file.a -fsanitize=fuzzer -fsanitize=integer -fsanitize-coverage=trace-cmp -fsanitize-coverage=inline-8bit-counters
clang -o fuzz_binary fuzz_archive_file.a -fsanitize=fuzzer

