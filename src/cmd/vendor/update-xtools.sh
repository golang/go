#!/bin/sh
#
# update-xtools.sh: idempotently update the vendored
# copy of the x/tools repository used by vet-lite.

set -u

analysis=$(go list -f {{.Dir}} golang.org/x/tools/go/analysis) ||
  { echo "Add golang.org/x/tools to your GOPATH"; exit 1; } >&2
xtools=$(dirname $(dirname $analysis))

vendor=$(dirname $0)

go list -f '{{.ImportPath}} {{.Dir}}' -deps golang.org/x/tools/go/analysis/cmd/vet-lite |
  grep golang.org/x/tools |
  while read path dir
  do
    mkdir -p $vendor/$path
    cp $dir/* -t $vendor/$path 2>/dev/null # ignore errors from subdirectories
    rm -f $vendor/$path/*_test.go
    git add $vendor/$path
  done

echo "Copied $xtools@$(cd $analysis && git rev-parse --short HEAD) to $vendor" >&2

go build -o /dev/null ./golang.org/x/tools/go/analysis/cmd/vet-lite ||
  { echo "Failed to build vet-lite"; exit 1; } >&2
