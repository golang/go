// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.16

package structtag_test

import (
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/structtag"
)

// Test the multiple key format added in Go 1.16.

func TestGo16(t *testing.T) {
	testdata := analysistest.TestData()
	analysistest.Run(t, testdata, structtag.Analyzer, "go16")
}
