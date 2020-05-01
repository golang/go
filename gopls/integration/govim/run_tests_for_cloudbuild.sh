#!/bin/bash

# Copyright 2020 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This script runs govim integration tests but always succeeds, instead writing
# their result to a file so that any test failure can be deferred to a later
# build step. We do this so that we can capture govim test artifacts regardless
# of the test results.

# Substitute the locally built gopls binary for use in govim integration tests.
go test -short ./cmd/govim -gopls /workspace/gopls/gopls

# Stash the error, for use in a later build step.
echo "exit $?" > /workspace/govim_test_result.sh

# Clean up unnecessary artifacts. This is based on govim/_scripts/tidyUp.bash.
# Since we're fetching govim using the go command, we won't have this non-go
# source directory available to us.
if [[ -n "$GOVIM_TESTSCRIPT_WORKDIR_ROOT" ]]; then
  echo "Cleaning up build artifacts..."
  # Make artifacts writable so that rm -rf doesn't complain.
  chmod -R u+w "$GOVIM_TESTSCRIPT_WORKDIR_ROOT"

  # Remove directories we don't care about.
  find "$GOVIM_TESTSCRIPT_WORKDIR_ROOT" -type d \( -name .vim -o -name gopath \) -prune -exec rm -rf '{}' \;
fi
