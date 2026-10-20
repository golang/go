// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"strings"
	"testing"
)

// M6-S8 scanner attacks: leading-dot continuation (SPEC T-104.6). In
// LogosMode a newline followed by blanks and then `.` plus a letter or
// `_` inserts no semicolon; `x\n.5` still does; plain mode is unchanged.

func scanTokens(t *testing.T, src string, mode uint) []token {
	t.Helper()
	var errs []string
	var s scanner
	s.init(strings.NewReader(src), func(_, _ uint, msg string) {
		errs = append(errs, msg)
	}, mode)
	var toks []token
	for {
		s.next()
		toks = append(toks, s.tok)
		if s.tok == _EOF {
			break
		}
	}
	if len(errs) != 0 {
		t.Fatalf("scan %q: errors %q", src, errs)
	}
	return toks
}

func hasSemi(toks []token) bool {
	for _, tok := range toks {
		if tok == _Semi {
			return true
		}
	}
	return false
}

func TestLeadingDotContinuation(t *testing.T) {
	for _, src := range []string{
		"batch\n.filter(x)\n",
		"batch\n    .filter(x)\n",
		"batch\n\t\t.map(f)\n",
		"batch\n\n\n.filter(x)\n",
		"batch\r\n.filter(x)\n",
		"batch\n._x\n",
		"batch\n.Filter(x)\n",
		"batch\n.éclair(x)\n",
	} {
		toks := scanTokens(t, src, logosScannerMode)
		// The token after the receiver name is the continuation dot:
		// no semicolon intervenes. (The final newline still yields
		// its own `;`, which the check below ignores.)
		if len(toks) < 2 || toks[0] != _Name || toks[1] != _Dot {
			t.Errorf("%q: got tokens %v; want _Name _Dot continuation", src, toks)
		}
	}
}

func TestLeadingDotKeepsSemi(t *testing.T) {
	for _, src := range []string{
		"x\n.5\n",
		"x\n.5e3\n",
		"x\n..y\n",
		"x\n. y\n",
		"x\n5\n",
		"x\n(1)\n",
	} {
		toks := scanTokens(t, src, logosScannerMode)
		// The newline still inserts `;` right after the receiver.
		if len(toks) < 2 || toks[0] != _Name || toks[1] != _Semi {
			t.Errorf("%q: got tokens %v; want _Name _Semi break", src, toks)
		}
	}
}

func TestLeadingDotPlainUnchanged(t *testing.T) {
	toks := scanTokens(t, "batch\n.filter(x)\n", 0)
	if !hasSemi(toks) {
		t.Errorf("plain scan: got no semicolon; want one")
	}
}

// TestLeadingDotParses checks the scanner-to-parser seam: a
// leading-dot continuation parses as one selector chain in LogosMode.
func TestLeadingDotParses(t *testing.T) {
	src := "package p\nfunc f(xs []int) {\n\t_ = xs\n\t\t.filter(func(x int) bool { return x > 0 })\n}\n"
	var msgs []string
	f, err := Parse(nil, strings.NewReader(src), func(err error) {
		msgs = append(msgs, err.Error())
	}, nil, CheckBranches|LogosMode)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(msgs) != 0 {
		t.Fatalf("parse errors %q", msgs)
	}
	found := false
	Inspect(f, func(n Node) bool {
		if sel, ok := n.(*SelectorExpr); ok && sel.Sel.Value == "filter" {
			found = true
		}
		return true
	})
	if !found {
		t.Fatalf("parsed tree holds no .filter selector")
	}
}
