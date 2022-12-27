// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nilness_test

import (
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/nilness"
	"golang.org/x/tools/internal/typeparams"
)

func Test(t *testing.T) {
	testdata := analysistest.TestData()
	analysistest.Run(t, testdata, nilness.Analyzer, "a")
}

func TestInstantiated(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("TestInstantiated requires type parameters")
	}
	testdata := analysistest.TestData()
	analysistest.Run(t, testdata, nilness.Analyzer, "c")
}

func TestTypeSet(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("TestTypeSet requires type parameters")
	}
	testdata := analysistest.TestData()
	analysistest.Run(t, testdata, nilness.Analyzer, "d")
}
