// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.17
// +build go1.17

package vta

import (
	"testing"

	"golang.org/x/tools/go/callgraph/cha"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

func TestVTACallGraphGo117(t *testing.T) {
	file := "testdata/src/go117.go"
	prog, want, err := testProg(file, ssa.BuilderMode(0))
	if err != nil {
		t.Fatalf("couldn't load test file '%s': %s", file, err)
	}
	if len(want) == 0 {
		t.Fatalf("couldn't find want in `%s`", file)
	}

	g, _ := typePropGraph(ssautil.AllFunctions(prog), cha.CallGraph(prog))
	if gs := vtaGraphStr(g); !subGraph(want, gs) {
		t.Errorf("`%s`: want superset of %v;\n got %v", file, want, gs)
	}
}
