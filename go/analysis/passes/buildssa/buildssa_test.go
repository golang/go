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
	"golang.org/x/tools/internal/typeparams"
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

func TestGenericDecls(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("TestGenericDecls requires type parameters.")
	}
	testdata := analysistest.TestData()
	result := analysistest.Run(t, testdata, buildssa.Analyzer, "b")[0].Result

	ssainfo := result.(*buildssa.SSA)
	got := fmt.Sprint(ssainfo.SrcFuncs)
	want := `[(*b.Pointer[T]).Load b.Load b.LoadPointer]`
	if got != want {
		t.Errorf("SSA.SrcFuncs = %s, want %s", got, want)
		for _, f := range ssainfo.SrcFuncs {
			f.WriteTo(os.Stderr)
		}
	}
}

func TestImporting(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("TestImporting depends on testdata/b/b/go which uses type parameters.")
	}
	testdata := analysistest.TestData()
	result := analysistest.Run(t, testdata, buildssa.Analyzer, "c")[0].Result

	ssainfo := result.(*buildssa.SSA)
	got := fmt.Sprint(ssainfo.SrcFuncs)
	want := `[c.A c.B]`
	if got != want {
		t.Errorf("SSA.SrcFuncs = %s, want %s", got, want)
		for _, f := range ssainfo.SrcFuncs {
			f.WriteTo(os.Stderr)
		}
	}
}
