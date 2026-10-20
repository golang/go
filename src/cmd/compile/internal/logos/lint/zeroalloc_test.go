// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"internal/testenv"
	"slices"
	"testing"
)

const zeroAllocFuncsSrc = `package p

type T struct{}

type G[K any] struct{}

//logos:zeroalloc
func plain() {}

// doc line
//logos:zeroalloc hot path
// another doc line
func (t *T) ptr() {}

//logos:zeroalloc
func (t T) val() {}

//logos:zeroalloc
func generic[K any](k K) K { return k }

//logos:zeroalloc
func (g *G[K]) m() {}

func unmarked() {}
`

// TestZeroAllocFuncs checks which functions a //logos:zeroalloc directive
// marks, in both kinds of package: a plain function, a method of either
// receiver kind, a generic function and a method of a generic type, a
// directive in the middle of a doc comment, and one followed by free text.
func TestZeroAllocFuncs(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	for name, kind := range pinKinds {
		t.Run(name, func(t *testing.T) {
			in := check(t, kind, name, zeroAllocFuncsSrc)
			got, misplaced := ZeroAllocFuncs(in)
			want := []ZeroAllocFunc{
				{Name: "plain", File: name, Line: 8},
				{Name: "(*T).ptr", File: name, Line: 13},
				{Name: "T.val", File: name, Line: 16},
				{Name: "generic", File: name, Line: 19},
				{Name: "(*G[K]).m", File: name, Line: 22},
			}
			if !slices.Equal(got, want) {
				t.Errorf("marked = %+v, want %+v", got, want)
			}
			if len(misplaced) != 0 {
				t.Errorf("misplaced = %q, want none", summarize(misplaced))
			}
		})
	}
}

const zeroAllocMisplacedSrc = `package p

//logos:zeroalloc

func detached() {}

//logos:zeroalloc
type T struct{}

func inBody() {
	//logos:zeroalloc
	_ = T{}
}

func trailing() {} //logos:zeroalloc

//logos:zeroalloc
func marked() {}
`

// TestZeroAllocMisplaced checks that a directive which marks nothing is an
// error at forbid, whatever the reason: a blank line before the function, a
// declaration that is not a function, a statement, and a comment after code.
// The directive that does precede a function still marks it.
func TestZeroAllocMisplaced(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	for name, kind := range pinKinds {
		t.Run(name, func(t *testing.T) {
			in := check(t, kind, name, zeroAllocMisplacedSrc)
			funcs, got := ZeroAllocFuncs(in)
			if want := []ZeroAllocFunc{{Name: "marked", File: name, Line: 18}}; !slices.Equal(funcs, want) {
				t.Errorf("marked = %+v, want %+v", funcs, want)
			}
			want := []string{
				"3 perf::zeroalloc_violation forbid",
				"7 perf::zeroalloc_violation forbid",
				"11 perf::zeroalloc_violation forbid",
				"15 perf::zeroalloc_violation forbid",
			}
			if sum := summarize(got); !slices.Equal(sum, want) {
				t.Errorf("misplaced = %q, want %q", sum, want)
			}
			const msg = "//logos:zeroalloc must be on its own line directly above a function declaration"
			for _, r := range got {
				if r.Message != msg {
					t.Errorf("message = %q, want %q", r.Message, msg)
				}
			}
		})
	}
}

const zeroAllocResultSrc = `package p

//logos:zeroalloc
func pack(v int) []int {
	//logos:allow(perf::zeroalloc_violation)
	return []int{v}
}
`

// TestZeroAllocResults covers the resolution of allocation sites: the
// message names the function and the reason, every result is at forbid, a
// reason repeated on one line (the wrapper of a generic function sits at
// another column) is reported once, and a site in a file the package does
// not hold is dropped. Neither a perf severity of allow nor a
// //logos:allow directive on the line changes the result.
func TestZeroAllocResults(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const reason = "[]int{...} escapes to heap: flows to result ~r0 (return)"
	p, err := ParsePolicy([]byte(`{"version":1,"groups":{"perf":"allow"}}`))
	if err != nil {
		t.Fatal(err)
	}
	for name, kind := range pinKinds {
		t.Run(name, func(t *testing.T) {
			in := check(t, kind, name, zeroAllocResultSrc)
			in.Policy = p
			got := ZeroAllocResults(in, []ZeroAllocFinding{
				{Func: "pack", File: name, Line: 6, Col: 13, Reason: reason},
				{Func: "pack", File: name, Line: 6, Col: 9, Reason: reason},
				{Func: "pack", File: "other.go", Line: 3, Col: 1, Reason: reason},
				{Func: "pack", File: name, Line: 6, Col: 9, Reason: "other reason"},
			})
			if len(got) != 2 {
				t.Fatalf("got %q, want two results", summarize(got))
			}
			r := got[0]
			if r.Rule != ZeroAllocRule || r.Severity != Forbid || r.Pos.Line() != 6 || r.Pos.Col() != 9 {
				t.Errorf("result = %+v, want %s at forbid on 6:9", r, ZeroAllocRule)
			}
			if want := "pack allocates: " + reason; r.Message != want {
				t.Errorf("message = %q, want %q", r.Message, want)
			}
			if got[1].Message != "pack allocates: other reason" {
				t.Errorf("second message = %q", got[1].Message)
			}
		})
	}
}

// TestZeroAllocNotConfigurable checks that the zeroalloc diagnostic is no
// registered lint rule, so no [lints] key can name it and the severity
// resolution of Run never sees it.
func TestZeroAllocNotConfigurable(t *testing.T) {
	if r, ok := Lookup(ZeroAllocRule); ok {
		t.Errorf("%s is registered as %+v", ZeroAllocRule, r)
	}
}
