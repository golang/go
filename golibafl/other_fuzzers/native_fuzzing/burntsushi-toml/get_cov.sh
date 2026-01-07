#!/bin/bash

# when running go test without -fuzz, go will only run the harness with the 
# provided seed entries. Seeds are manually added in the code via
# f.Add. Furthermore, go takes all seeds in testdata/fuzz/<fuzz_function_name>
go test -tags gocov -run=FuzzMe -cover -coverpkg=$(go list -deps -test| grep "github.com/BurntSushi/" |  tr '\n' ',') -coverprofile cover.out
go tool cover -html cover.out -o cover.html
