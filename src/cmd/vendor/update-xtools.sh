#!/bin/sh
#
# update-xtools.sh: idempotently update the vendored
# copy of the x/tools repository used by cmd/vet.

set -u

analysis=$(go list -f {{.Dir}} golang.org/x/tools/go/analysis) ||
  { echo "Add golang.org/x/tools to your GOPATH"; exit 1; } >&2
xtools=$(dirname $(dirname $analysis))

vendor=$(dirname $0)

# Find the x/tools packages directly imported by cmd/vet.
go list -f '{{range $k, $v := .ImportMap}}{{$k}} {{end}}' cmd/vet |
  grep golang.org/x/tools |
  # Vendor their transitive closure of dependencies.
  xargs go list -f '{{.ImportPath}} {{.Dir}}' -deps |
  grep golang.org/x/tools |
  while read path dir
  do
    mkdir -p $vendor/$path
    cp $dir/* -t $vendor/$path 2>/dev/null # ignore errors from subdirectories
    rm -f $vendor/$path/*_test.go
    git add $vendor/$path
  done

echo "Copied $xtools@$(cd $analysis && git rev-parse --short HEAD) to $vendor" >&2

go build -o /dev/null cmd/vet ||
  { echo "Failed to build cmd/vet"; exit 1; } >&2
