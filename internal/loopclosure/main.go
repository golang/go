// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The loopclosure command applies the golang.org/x/tools/go/analysis/passes/loopclosure
// analysis to the specified packages of Go source code. It enables
// experimental checking of parallel subtests.
//
// TODO: Once the parallel subtest experiment is complete, this can be made
// public at go/analysis/passes/loopclosure/cmd, or deleted.
package main

import (
	"golang.org/x/tools/go/analysis/passes/loopclosure"
	"golang.org/x/tools/go/analysis/singlechecker"
	"golang.org/x/tools/internal/analysisinternal"
)

func main() {
	analysisinternal.LoopclosureParallelSubtests = true
	singlechecker.Main(loopclosure.Analyzer)
}
