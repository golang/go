#!/bin/bash

# get caddy specific coverage
go test -tags gocov -run=FuzzMe -cover -coverpkg=$(go list -deps -test| grep "github.com/caddy" |  tr '\n' ',') -coverprofile cover.out
go tool cover -html cover.out -o cover.html
