// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"testing"
)

// TestBinarySearchFuncInlinable checks that BinarySearchFunc stays within the
// inliner's budget. Inlining it is what lets the calls to cmp be made directly
// instead of through the func value, which is worth considerably more than the
// code it costs: while BinarySearchFunc was over budget, searching with a
// comparison function was substantially slower than the equivalent
// sort.Search.
//
// The check cannot be made by compiling package slices itself, since the body
// of a generic function is only checked for inlinability once instantiated, so
// instantiate it in a package of our own and compile that.
func TestBinarySearchFuncInlinable(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	dir := t.TempDir()
	src := filepath.Join(dir, "x.go")
	const prog = `package x

import "slices"

var _ = slices.BinarySearchFunc[[]string, string, string]
`
	if err := os.WriteFile(src, []byte(prog), 0o644); err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-gcflags=-m", src)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v: %v\n%s", cmd, err, out)
	}

	inlCands := collectInlCands(string(out))
	want := "slices.BinarySearchFunc[go.shape.[]string,go.shape.string,go.shape.string]"
	if _, ok := inlCands[want]; !ok {
		t.Errorf("BinarySearchFunc is no longer inlinable: %q not found in compiler output:\n%s", want, out)
	}
}
