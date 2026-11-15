// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"strings"
	"testing"
)

type tokRec struct {
	tok  token
	op   Operator
	kind LitKind
}

func scanTokSeq(t *testing.T, src string, mode uint) ([]tokRec, []string) {
	t.Helper()
	var toks []tokRec
	var errs []string
	var s scanner
	s.init(strings.NewReader(src), func(_, _ uint, msg string) {
		errs = append(errs, msg)
	}, mode)
	for range 16 {
		s.next()
		toks = append(toks, tokRec{s.tok, s.op, s.kind})
		if s.tok == _EOF {
			break
		}
	}
	return toks, errs
}

func tokensEqual(t *testing.T, src string, got, want []tokRec) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%q: got %v; want %v", src, got, want)
	}
	for i := range got {
		if got[i].tok != want[i].tok {
			t.Fatalf("%q: token %d: got %v; want %v", src, i, got[i], want[i])
		}
		if got[i].tok == _Operator && got[i].op != want[i].op {
			t.Fatalf("%q: token %d: got %v; want %v", src, i, got[i], want[i])
		}
		if got[i].tok == _Literal && got[i].kind != want[i].kind {
			t.Fatalf("%q: token %d: got %v; want %v", src, i, got[i], want[i])
		}
	}
}

func TestLogosFatArrowScan(t *testing.T) {
	toks, errs := scanTokSeq(t, "a=>b", logosScannerMode)
	tokensEqual(t, "a=>b", toks, []tokRec{
		{tok: _Name}, {tok: _FatArrow}, {tok: _Name}, {tok: _Semi}, {tok: _EOF},
	})
	if len(errs) != 0 {
		t.Fatalf("a=>b: got errors %q; want none", errs)
	}

	toks, _ = scanTokSeq(t, "a = > b", logosScannerMode)
	tokensEqual(t, "a = > b", toks, []tokRec{
		{tok: _Name}, {tok: _Assign}, {tok: _Operator, op: Gtr}, {tok: _Name}, {tok: _Semi}, {tok: _EOF},
	})

	toks, _ = scanTokSeq(t, "a=\n>b", logosScannerMode)
	tokensEqual(t, "a=\\n>b", toks, []tokRec{
		{tok: _Name}, {tok: _Assign}, {tok: _Operator, op: Gtr}, {tok: _Name}, {tok: _Semi}, {tok: _EOF},
	})

	toks, _ = scanTokSeq(t, "a=>b", 0)
	tokensEqual(t, "plain a=>b", toks, []tokRec{
		{tok: _Name}, {tok: _Assign}, {tok: _Operator, op: Gtr}, {tok: _Name}, {tok: _Semi}, {tok: _EOF},
	})
}

func TestLogosInterpScan(t *testing.T) {
	toks, errs := scanTokSeq(t, `x := $"a{b}c"`, logosScannerMode)
	tokensEqual(t, "interp", toks, []tokRec{
		{tok: _Name}, {tok: _Define}, {tok: _Literal, kind: InterpLit}, {tok: _Semi}, {tok: _EOF},
	})
	if len(errs) != 0 {
		t.Fatalf("interp: got errors %q; want none", errs)
	}

	toks, errs = scanTokSeq(t, `x := $ "a{b}c"`, logosScannerMode)
	if len(toks) > 2 && toks[2].tok == _Literal && toks[2].kind == InterpLit {
		t.Fatalf("gap form scanned as an interpolated literal: %v", toks)
	}
	if len(errs) == 0 {
		t.Fatalf("gap form: got no errors; want invalid character")
	}

	toks, errs = scanTokSeq(t, `x := $"a"`, 0)
	if len(toks) > 2 && toks[2].tok == _Literal && toks[2].kind == InterpLit {
		t.Fatalf("plain mode scanned an interpolated literal: %v", toks)
	}
	if len(errs) == 0 {
		t.Fatalf("plain mode: got no errors; want invalid character")
	}
}
