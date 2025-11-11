// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type T struct {
	x [2]int64 // field that will be clobbered. Also makes type not SSAable.
	p *byte    // has a pointer
}

//go:noinline
func makeT() T {
	return T{}
}

var g T

var sink any

func TestIssue15854(t *testing.T) {
	for i := 0; i < 10000; i++ {
		if g.x[0] != 0 {
			t.Fatalf("g.x[0] clobbered with %x\n", g.x[0])
		}
		// The bug was in the following assignment. The return
		// value of makeT() is not copied out of the args area of
		// stack frame in a timely fashion. So when write barriers
		// are enabled, the marshaling of the args for the write
		// barrier call clobbers the result of makeT() before it is
		// read by the write barrier code.
		g = makeT()
		sink = make([]byte, 1000) // force write barriers to eventually happen
	}
}
func TestIssue15854b(t *testing.T) {
	const N = 10000
	a := make([]T, N)
	for i := 0; i < N; i++ {
		a = append(a, makeT())
		sink = make([]byte, 1000) // force write barriers to eventually happen
	}
	for i, v := range a {
		if v.x[0] != 0 {
			t.Fatalf("a[%d].x[0] clobbered with %x\n", i, v.x[0])
		}
	}
}

// Test that the generated assembly has line numbers (Issue #16214).
func TestIssue16214(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()

	src := filepath.Join(dir, "x.go")
	err := os.WriteFile(src, []byte(issue16214src), 0644)
	if err != nil {
		t.Fatalf("could not write file: %v", err)
	}

	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-p=main", "-S", "-o", filepath.Join(dir, "out.o"), src)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go tool compile: %v\n%s", err, out)
	}

	if strings.Contains(string(out), "unknown line number") {
		t.Errorf("line number missing in assembly:\n%s", out)
	}
}

var issue16214src = `
package main

func Mod32(x uint32) uint32 {
	return x % 3 // frontend rewrites it as HMUL with 2863311531, the LITERAL node has unknown Pos
}
`
