#!/bin/bash

find ./internal/lsp/ -name *.golden -delete
go test ./internal/lsp/source  -golden
go test ./internal/lsp/ -golden
go test ./internal/lsp/cmd  -golden
