#!/bin/bash

# loop because go-test will stop when there is a panic
while true; do
    go test -fuzz FuzzMe -test.fuzzcachedir=./testdata/fuzz -parallel=1
    sleep 1
done