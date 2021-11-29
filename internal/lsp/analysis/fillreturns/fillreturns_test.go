// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fillreturns_test

import (
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/internal/lsp/analysis/fillreturns"
	"golang.org/x/tools/internal/typeparams"
)

func Test(t *testing.T) {
	t.Skip("temporarily skipped until CL 367196 is submitted and this test is adjusted accordingly")
	testdata := analysistest.TestData()
	tests := []string{"a"}
	if typeparams.Enabled {
		tests = append(tests, "typeparams")
	}
	analysistest.RunWithSuggestedFixes(t, testdata, fillreturns.Analyzer, tests...)
}
