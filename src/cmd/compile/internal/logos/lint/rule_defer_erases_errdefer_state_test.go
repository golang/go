// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"fmt"
	"slices"
	"strings"
	"testing"
)

const deferErasesSrc = `package p

import "errors"

var failure = errors.New("failed")

func rollback() {}

func after() (err error) {
	errdefer rollback()
	defer func() {
		err = nil // want
	}()
	return failure
}

func before() (err error) {
	defer func() {
		err = nil // want
	}()
	errdefer rollback()
	return failure
}

func branch() (err error) {
	errdefer rollback()
	defer func() {
		if recover() != nil {
			err = nil // want
		}
	}()
	return failure
}

func tuple() (n int, err error) {
	errdefer rollback()
	defer func() {
		n, err = 7, nil // want
	}()
	return 1, failure
}

func closure() func() error {
	return func() (err error) {
		errdefer rollback()
		defer func() {
			err = nil // want
		}()
		return failure
	}
}

func noErrdefer() (err error) {
	defer func() {
		err = nil
	}()
	return failure
}

func errdeferInClosureOnly() (err error) {
	run := func() (inner error) {
		errdefer rollback()
		return failure
	}
	defer func() {
		err = nil
	}()
	return run()
}

func notNil() (err error) {
	errdefer rollback()
	defer func() {
		err = errors.New("wrapped")
	}()
	return failure
}

func otherResult() (n int, err error) {
	errdefer rollback()
	defer func() {
		n = 0
	}()
	return 1, failure
}

func shadowed() (err error) {
	errdefer rollback()
	defer func() {
		var err error
		err = nil
		_ = err
	}()
	return failure
}

func nestedLiteral() (err error) {
	errdefer rollback()
	defer func() {
		forget := func() { err = nil }
		_ = forget
	}()
	return failure
}

func namedCallee() (err error) {
	errdefer rollback()
	defer rollback()
	return failure
}

func unnamed() error {
	errdefer rollback()
	defer func() {
		_ = failure
	}()
	return failure
}

func tryDefer() (err error) {
	errdefer rollback()
	defer func() error { return nil }()?
	return failure
}

func scoped() (err error) {
	errdefer rollback()
	{
		scope defer func() {
			err = nil
		}()
	}
	return failure
}
`

// TestDeferErasesErrdeferState covers the detection and the exemptions of
// correctness::defer_erases_errdefer_state.
func TestDeferErasesErrdeferState(t *testing.T) {
	expect(t, KindLgo, nil, testSource{"x.lgo", deferErasesSrc}, finding{"correctness::defer_erases_errdefer_state", Forbid})
}

// TestDeferErasesErasureSpan checks the span of a finding: the assignment,
// +// which tools splice by this byte range.
func TestDeferErasesErasureSpan(t *testing.T) {
	const src = `package p

func rollback() {}

func f() (failed error) {
	errdefer rollback()
	defer func() {
		failed = nil
	}()
	return nil
}
`
	got := Run(check(t, KindLgo, "x.lgo", src))
	if len(got) != 1 {
		t.Fatalf("results %q, want one", summarize(got))
	}
	if text := spanText(src, got[0]); text != "failed = nil" {
		t.Errorf("span %q, want %q", text, "failed = nil")
	}
}

func TestDeferErasesErrdeferStateForbid(t *testing.T) {
	src := strings.Replace(deferErasesSrc, "func after() (err error) {", "//logos:allow(correctness::defer_erases_errdefer_state)\nfunc after() (err error) {", 1)
	got := summarize(Run(check(t, KindLgo, "x.lgo", src)))
	want := wantMarked(src, "correctness::defer_erases_errdefer_state", Forbid)
	directive := 0
	for i, line := range strings.Split(src, "\n") {
		if strings.Contains(line, "logos:allow(correctness::defer_erases_errdefer_state)") {
			directive = i + 1
		}
	}
	if directive == 0 {
		t.Fatal("allow directive lost from source")
	}
	want = append(want, fmt.Sprintf("%d %s %s", directive, AllowForbiddenRule, Forbid))
	slices.Sort(got)
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Errorf("results = %q, want %q", got, want)
	}
}
