#!/bin/bash
# Copyright 2015 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This script regenerates deps.go.
# The script used to do all the work, but now a Go program does.
# The script has been preserved so that people who learned to type
# ./mkdeps.bash don't have to relearn a new method.
# It's fine to run "go run mkdeps.go" directly instead.

set -e
go run mkdeps.go -- "$@"
exit 0
