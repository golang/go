#!/bin/bash -e

# Copyright 2019 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Run govim integration tests against a local gopls.
# TODO(findleyr): this script assumes that docker may be run without sudo.
# Update it to escalate privileges if and only if necessary.

# Find the tools root, so that this script can be run from any directory.
script_dir=$(dirname "$(readlink -f "$0")")
tools_dir=$(readlink -f "${script_dir}/../../..")

# Build gopls.
cd "${tools_dir}/gopls"
temp_gopls=$(mktemp -p "$PWD")
trap "rm -f \"${temp_gopls}\"" EXIT
go build -o "${temp_gopls}"

# Build the test harness. Here we are careful to pass in a very limited build
# context so as to optimize caching.
cd "${tools_dir}"
docker build -t gopls-govim-harness -f gopls/integration/govim/Dockerfile \
  gopls/integration/govim

# Run govim integration tests.
echo "running govim integration tests using ${temp_gopls}"
temp_gopls_name=$(basename "${temp_gopls}")
docker run --rm -t \
  -v "${tools_dir}:/src/tools" \
  -w "/src/govim" \
  gopls-govim-harness \
  go test ./cmd/govim \
    -gopls "/src/tools/gopls/${temp_gopls_name}"
