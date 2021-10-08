// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loopclosure_test

import (
	"golang.org/x/tools/internal/typeparams"
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/loopclosure"
)

func Test(t *testing.T) {
	testdata := analysistest.TestData()
	tests := []string{"a", "golang.org/..."}
	if typeparams.Enabled {
		tests = append(tests, "typeparams")
	}
	analysistest.Run(t, testdata, loopclosure.Analyzer, tests...)
}
