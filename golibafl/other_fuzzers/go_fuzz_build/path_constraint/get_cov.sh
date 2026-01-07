#!/bin/bash

# get caddy specific coverage
go test -tags gocov -run=FuzzMe -cover -coverpkg=srlabs.de/harness -coverprofile cover.out
go tool cover -html cover.out -o cover.html

