// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"strings"
	"testing"
)

func TestPlainGoRejectsConstPositions(t *testing.T) {
	for _, src := range []string{
		"package p\n\nconst for i := range x {}",
		"package p\n\nconst if true {}",
		"package p\n\nconst switch x {\n}",
		"package p\n\nconst func F() {}",
		"package p\n\ntype I interface {\n\tconst M() int\n}",
		"package p\n\ntype I interface {\n\tconst Name string\n}",
		"package p\n\nfunc f(x const []int) {}",
		"package p\n\nvar v const int",
	} {
		var errs []string
		_, _ = Parse(nil, strings.NewReader(src), func(err error) {
			errs = append(errs, err.Error())
		}, nil, 0)
		if len(errs) == 0 {
			t.Errorf("%q: parsed without error; want a syntax error", src)
		}
	}
}

func TestLogosConstUnlistedPositionsReject(t *testing.T) {
	tests := []struct {
		name string
		src  string
		pos  string
	}{
		{"struct field", "package p\ntype S struct {\n\tconst F int\n}\n", ":3:2"},
		{"parameter name", "package p\nfunc f(const x int) {}\n", ":2:16"},
		{"bare statement", "package p\nfunc f() {\n\tconst\n}\n", ":4:1"},
		{"expression", "package p\nfunc f() {\n\t_ = const\n}\n", ":4:1"},
		{"top level method", "package p\nfunc f() {}\nconst M() int\n", ":3:11"},
	}
	for _, test := range tests {
		logos := parseErrs(t, test.src, CheckBranches|LogosMode)
		if len(logos) == 0 {
			t.Errorf("%s: logos mode parsed without error; want a syntax error at %s", test.name, test.pos)
			continue
		}
		if !strings.Contains(logos[0], "syntax error") || !strings.Contains(logos[0], test.pos) {
			t.Errorf("%s: logos error %q names no syntax error at %s", test.name, logos[0], test.pos)
		}
		plain := parseErrs(t, test.src, CheckBranches)
		if len(plain) == 0 {
			t.Errorf("%s: plain mode parsed without error; want a syntax error as in plain Go", test.name)
		}
	}
}
