// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildssa_test

import (
	"fmt"
	"os"
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/buildssa"
)

func Test(t *testing.T) {
	testdata := analysistest.TestData()
	result := analysistest.Run(t, testdata, buildssa.Analyzer, "a")[0].Result

	ssainfo := result.(*buildssa.SSA)
	got := fmt.Sprint(ssainfo.SrcFuncs)
	want := `[a.Fib (a.T).fib]`
	if got != want {
		t.Errorf("SSA.SrcFuncs = %s, want %s", got, want)
		for _, f := range ssainfo.SrcFuncs {
			f.WriteTo(os.Stderr)
		}
	}
}
