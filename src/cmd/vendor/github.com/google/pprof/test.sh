#!/usr/bin/env bash

set -e
MODE=atomic
echo "mode: $MODE" > coverage.txt

PKG=$(go list ./... | grep -v /vendor/)

staticcheck $PKG
unused $PKG
go test -v $PKG

for d in $PKG; do
  go test -race -coverprofile=profile.out -covermode=$MODE $d
  if [ -f profile.out ]; then
    cat profile.out | grep -v "^mode: " >> coverage.txt
    rm profile.out
  fi
done
