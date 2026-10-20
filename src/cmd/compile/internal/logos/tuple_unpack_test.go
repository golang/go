// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"regexp"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
)

var tupleTempName = regexp.MustCompile(`_logos_tup[0-9]+_[0-9]+_0`)

// requirePlainGo fails the test unless the ejected text parses as plain
// Go: a parenthesized comma list or a dangling `)` left behind by a
// lowering edit is a syntax error there.
func requirePlainGo(t *testing.T, ejected string) {
	t.Helper()
	var errs []error
	_, err := syntax.Parse(nil, strings.NewReader(ejected), func(err error) { errs = append(errs, err) }, nil, 0)
	if err != nil || len(errs) != 0 {
		t.Errorf("ejected text is not plain Go: %v %v\n%s", err, errs, ejected)
	}
}

// TestTupleUnpackParenOperand pins the spelled-out unpack of a
// parenthesized tuple operand: every redundant pair around the source
// goes with it, so `u, v := ((t))` lowers to `u, v := t.T_0, t.T_1` and
// never leaves a parenthesized comma list behind (invalid Go, and a
// *TupleLit when re-parsed).
func TestTupleUnpackParenOperand(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"define bare", "u, v := t", "u, v := t.T_0, t.T_1"},
		{"define one pair", "u, v := (t)", "u, v := t.T_0, t.T_1"},
		{"define two pairs", "u, v := ((t))", "u, v := t.T_0, t.T_1"},
		{"define spaced pairs", "u, v := ( ( t ) )", "u, v := t.T_0, t.T_1"},
		{"assign bare", "var u, v int\n\tu, v = t", "u, v = t.T_0, t.T_1"},
		{"assign one pair", "var u, v int\n\tu, v = (t)", "u, v = t.T_0, t.T_1"},
		{"assign two pairs", "var u, v int\n\tu, v = ((t))", "u, v = t.T_0, t.T_1"},
		{"assign spaced pairs", "var u, v int\n\tu, v = ( ( t ) )", "u, v = t.T_0, t.T_1"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			src := "package main\n\nfunc f() {\n\tt := (1, 2)\n\t" + c.body + "\n\t_, _ = u, v\n}\n"
			_, ejected := lowerSpreadSrc(t, src)
			if !strings.Contains(ejected, "\t"+c.want+"\n") {
				t.Errorf("ejected text lacks %q:\n%s", c.want, ejected)
			}
			requirePlainGo(t, ejected)
		})
	}
}

// TestTupleUnpackParenSelector pins the same rule when the operand is
// not a bare name: the source is bound once, and the wrapping pairs go
// with the replaced span instead of leaving a dangling `)`.
func TestTupleUnpackParenSelector(t *testing.T) {
	cases := []struct {
		name string
		body string
		op   string
	}{
		{"define bare", "u, v := s.t", ":="},
		{"define one pair", "u, v := (s.t)", ":="},
		{"define two pairs", "u, v := ((s.t))", ":="},
		{"assign bare", "var u, v int\n\tu, v = s.t", "="},
		{"assign one pair", "var u, v int\n\tu, v = (s.t)", "="},
		{"assign two pairs", "var u, v int\n\tu, v = ((s.t))", "="},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			src := "package main\n\ntype S struct{ t (int, int) }\n\nfunc f(s S) {\n\t" + c.body + "\n\t_, _ = u, v\n}\n"
			_, ejected := lowerSpreadSrc(t, src)
			tmp := tupleTempName.FindString(ejected)
			want := "\t" + tmp + " := s.t; u, v " + c.op + " " + tmp + ".T_0, " + tmp + ".T_1\n"
			if tmp == "" || !strings.Contains(ejected, want) {
				t.Errorf("ejected text lacks %q:\n%s", want, ejected)
			}
			requirePlainGo(t, ejected)
		})
	}
}

// TestTupleReturnParenOperand pins the same rule for `return ((t))`,
// which spells its reads through the same paren-pair extension.
func TestTupleReturnParenOperand(t *testing.T) {
	for _, operand := range []string{"t", "(t)", "((t))", "( ( t ) )"} {
		t.Run(operand, func(t *testing.T) {
			src := "package main\n\nfunc g(t (int, int)) (int, int) {\n\treturn " + operand + "\n}\n"
			_, ejected := lowerSpreadSrc(t, src)
			if !strings.Contains(ejected, "\treturn t.T_0, t.T_1\n") {
				t.Errorf("ejected text lacks %q:\n%s", "return t.T_0, t.T_1", ejected)
			}
			requirePlainGo(t, ejected)
		})
	}
}

// TestTupleReturnParenSelector pins `return (s.t)`: the source is bound
// once and the wrapping pairs leave with the reads.
func TestTupleReturnParenSelector(t *testing.T) {
	for _, operand := range []string{"s.t", "(s.t)", "((s.t))"} {
		t.Run(operand, func(t *testing.T) {
			src := "package main\n\ntype S struct{ t (int, int) }\n\nfunc g(s S) (int, int) {\n\treturn " + operand + "\n}\n"
			_, ejected := lowerSpreadSrc(t, src)
			tmp := tupleTempName.FindString(ejected)
			want := "\t" + tmp + " := s.t; return " + tmp + ".T_0, " + tmp + ".T_1\n"
			if tmp == "" || !strings.Contains(ejected, want) {
				t.Errorf("ejected text lacks %q:\n%s", want, ejected)
			}
			requirePlainGo(t, ejected)
		})
	}
}
