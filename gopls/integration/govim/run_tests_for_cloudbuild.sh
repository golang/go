#!/bin/bash

# Copyright 2020 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This script runs govim integration tests but always succeeds, instead writing
# their result to a file so that any test failure can be deferred to a later
# build step. We do this so that we can capture govim test artifacts regardless
# of the test results.

# Substitute the locally built gopls binary for use in govim integration tests.
go test ./cmd/govim -gopls /workspace/gopls/gopls

# Stash the error, for use in a later build step.
echo "exit $?" > /workspace/govim_test_result.sh
