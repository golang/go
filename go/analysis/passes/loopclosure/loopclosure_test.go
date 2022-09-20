// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loopclosure_test

import (
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/loopclosure"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/typeparams"
)

func Test(t *testing.T) {
	testdata := analysistest.TestData()
	tests := []string{"a", "golang.org/..."}
	if typeparams.Enabled {
		tests = append(tests, "typeparams")
	}
	analysistest.Run(t, testdata, loopclosure.Analyzer, tests...)

	// Enable checking of parallel subtests.
	defer func(parallelSubtest bool) {
		analysisinternal.LoopclosureParallelSubtests = parallelSubtest
	}(analysisinternal.LoopclosureParallelSubtests)
	analysisinternal.LoopclosureParallelSubtests = true

	analysistest.Run(t, testdata, loopclosure.Analyzer, "subtests")
}
