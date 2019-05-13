// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt_test

import (
	"errors"
	"fmt"
	"testing"
)

func TestErrorf(t *testing.T) {
	wrapped := errors.New("inner error")
	for _, test := range []struct {
		err        error
		wantText   string
		wantUnwrap error
	}{{
		err:        fmt.Errorf("%w", wrapped),
		wantText:   "inner error",
		wantUnwrap: wrapped,
	}, {
		err:        fmt.Errorf("added context: %w", wrapped),
		wantText:   "added context: inner error",
		wantUnwrap: wrapped,
	}, {
		err:        fmt.Errorf("%w with added context", wrapped),
		wantText:   "inner error with added context",
		wantUnwrap: wrapped,
	}, {
		err:        fmt.Errorf("%s %w %v", "prefix", wrapped, "suffix"),
		wantText:   "prefix inner error suffix",
		wantUnwrap: wrapped,
	}, {
		err:        fmt.Errorf("%[2]s: %[1]w", wrapped, "positional verb"),
		wantText:   "positional verb: inner error",
		wantUnwrap: wrapped,
	}, {
		err:      fmt.Errorf("%v", wrapped),
		wantText: "inner error",
	}, {
		err:      fmt.Errorf("added context: %v", wrapped),
		wantText: "added context: inner error",
	}, {
		err:      fmt.Errorf("%v with added context", wrapped),
		wantText: "inner error with added context",
	}, {
		err:      fmt.Errorf("%w is not an error", "not-an-error"),
		wantText: "%!w(string=not-an-error) is not an error",
	}, {
		err:      fmt.Errorf("wrapped two errors: %w %w", errString("1"), errString("2")),
		wantText: "wrapped two errors: 1 %!w(fmt_test.errString=2)",
	}, {
		err:      fmt.Errorf("wrapped three errors: %w %w %w", errString("1"), errString("2"), errString("3")),
		wantText: "wrapped three errors: 1 %!w(fmt_test.errString=2) %!w(fmt_test.errString=3)",
	}, {
		err:        fmt.Errorf("%w", nil),
		wantText:   "%!w(<nil>)",
		wantUnwrap: nil, // still nil
	}} {
		if got, want := errors.Unwrap(test.err), test.wantUnwrap; got != want {
			t.Errorf("Formatted error: %v\nerrors.Unwrap() = %v, want %v", test.err, got, want)
		}
		if got, want := test.err.Error(), test.wantText; got != want {
			t.Errorf("err.Error() = %q, want %q", got, want)
		}
	}
}

type errString string

func (e errString) Error() string { return string(e) }
