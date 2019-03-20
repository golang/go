// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tests_test

import (
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/tests"
)

func Test(t *testing.T) {
	testdata := analysistest.TestData()

	analysistest.Run(t, testdata, tests.Analyzer,
		"a",        // loads "a", "a [a.test]", and "a.test"
		"b_x_test", // loads "b" and "b_x_test"
		"divergent",
	)
}
