#!/bin/bash -e

# Copyright 2019 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Run govim integration tests against a local gopls.

usage() {
  cat <<EOUSAGE
Usage: $0 [--sudo] [--short] [--version (semver|latest)]

Args:
  --sudo     run docker with sudo
  --short    run `go test` with `-short`
  --version  run on the specific tagged govim version (or latest) rather
             than the default branch

Run govim tests against HEAD using local docker.
EOUSAGE
}

SUDO_IF_NEEDED=
TEST_SHORT=
DOCKERFILE=gopls/integration/govim/Dockerfile
GOVIM_REF=main
while [[ $# -gt 0 ]]; do
  case "$1" in
    "-h" | "--help" | "help")
      usage
      exit 0
      ;;
    "--sudo")
      SUDO_IF_NEEDED="sudo "
      shift
      ;;
    "--short")
      TEST_SHORT="-short"
      shift
      ;;
    "--version")
      if [[ -z "$2" ]]; then
        usage
        exit 1
      fi
      GOVIM_REF=$2
      if [[ "${GOVIM_REF}" == "latest" ]]; then
        TMPGOPATH=$(mktemp -d)
        trap "GOPATH=${TMPGOPATH} go clean -modcache && rm -r ${TMPGOPATH}" EXIT
        GOVIM_REF=$(GOPATH=${TMPGOPATH} go mod download -json \
          github.com/govim/govim@latest | jq -r .Version)
      fi
      shift 2
      ;;
    *)
      usage
      exit 1
  esac
done

# Find the tools root, so that this script can be run from any directory.
script_dir=$(dirname "$(readlink -f "$0")")
tools_dir=$(readlink -f "${script_dir}/../../..")

# Build gopls.
cd "${tools_dir}/gopls"
temp_gopls=$(mktemp -p "$PWD")
trap "rm -f \"${temp_gopls}\"" EXIT
# For consistency across environments, use golang docker to build rather than
# the local go command.
${SUDO_IF_NEEDED}docker run --rm -t \
  -v "${tools_dir}:/src/tools" \
  -w "/src/tools/gopls" \
  golang:rc \
  go build -o $(basename ${temp_gopls})

# Build the test harness. Here we are careful to pass in a very limited build
# context so as to optimize caching.
echo "Checking out govim@${GOVIM_REF}"
cd "${tools_dir}"
${SUDO_IF_NEEDED}docker build \
  --build-arg GOVIM_REF="${GOVIM_REF}" \
  -t gopls-govim-harness:${GOVIM_REF} \
  -f gopls/integration/govim/Dockerfile \
  gopls/integration/govim

# Run govim integration tests.
echo "running govim integration tests using ${temp_gopls}"
temp_gopls_name=$(basename "${temp_gopls}")
${SUDO_IF_NEEDED}docker run --rm -t \
  -v "${tools_dir}:/src/tools" \
  -w "/src/govim" \
  --ulimit memlock=-1:-1 \
  gopls-govim-harness:${GOVIM_REF} \
  go test ${TEST_SHORT} ./cmd/govim \
    -gopls "/src/tools/gopls/${temp_gopls_name}"
