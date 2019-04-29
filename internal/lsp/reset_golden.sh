#!/bin/bash

find ./internal/lsp/ -name *.golden -delete
go test ./internal/lsp/ ./internal/lsp/cmd ./internal/lsp/source -golden
