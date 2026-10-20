// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"strings"
	"testing"
)

// M6-S4 parser attacks for SpreadExpr{X Expr}
// (logos-docs/IMPLEMENTATION.md): `..src` parses as a spread element in
// LogosMode only; plain mode keeps the upstream missing-operand error.

func parseSpread(t *testing.T, src string) *File {
	t.Helper()
	var msgs []string
	f, err := Parse(nil, strings.NewReader(src), func(err error) {
		msgs = append(msgs, err.Error())
	}, nil, CheckBranches|LogosMode)
	if err != nil {
		t.Fatalf("parse %q: %v", src, err)
	}
	if len(msgs) != 0 {
		t.Fatalf("parse %q: errors %q", src, msgs)
	}
	return f
}

func TestSpreadParsesAsElement(t *testing.T) {
	f := parseSpread(t, "package p\ntype S struct{ A, B int }\nvar _ = S{A: 1, ..s}\n")
	lit := f.DeclList[1].(*VarDecl).Values.(*CompositeLit)
	if len(lit.ElemList) != 2 {
		t.Fatalf("got %d elements; want 2", len(lit.ElemList))
	}
	sp, ok := lit.ElemList[1].(*SpreadExpr)
	if !ok {
		t.Fatalf("element 1 is %T; want *SpreadExpr", lit.ElemList[1])
	}
	if _, ok := sp.X.(*Name); !ok {
		t.Fatalf("spread source is %T; want *Name", sp.X)
	}
	if got := StartPos(sp); got != sp.Pos() {
		t.Fatalf("StartPos is %v; want node pos %v", got, sp.Pos())
	}
}

func fullSpread(t *testing.T, f *File) string {
	t.Helper()
	var buf strings.Builder
	// Form 0 is the full form: unlike String (ShortForm), it keeps
	// composite literal contents instead of printing `{…}`.
	if _, err := Fprint(&buf, f, 0); err != nil {
		t.Fatalf("print: %v", err)
	}
	return buf.String()
}

func TestSpreadPrinterRoundTrip(t *testing.T) {
	src := "package p\ntype S struct{ A, B int }\nvar _ = S{A: s.B, B: s.A, ..s}\n"
	f := parseSpread(t, src)
	if out := fullSpread(t, f); !strings.Contains(out, "..s") {
		t.Fatalf("printed output misses spread: %q", out)
	}
	g := parseSpread(t, fullSpread(t, f))
	if fullSpread(t, g) != fullSpread(t, f) {
		t.Fatalf("reprint differs:\n%s\n%s", fullSpread(t, f), fullSpread(t, g))
	}
}

func TestSpreadWalkVisitsSource(t *testing.T) {
	f := parseSpread(t, "package p\ntype S struct{ A int }\nvar _ = S{..s}\n")
	var names []string
	Inspect(f, func(n Node) bool {
		if sp, ok := n.(*SpreadExpr); ok {
			if name, ok := sp.X.(*Name); ok {
				names = append(names, name.Value)
			}
		}
		return true
	})
	if len(names) != 1 || names[0] != "s" {
		t.Fatalf("walk saw sources %q; want [s]", names)
	}
}

func TestSpreadPlainModeUnchanged(t *testing.T) {
	src := "package p\ntype S struct{ A int }\nvar _ = S{..s}\n"
	var msgs []string
	_, err := Parse(nil, strings.NewReader(src), func(err error) {
		msgs = append(msgs, err.Error())
	}, nil, CheckBranches)
	if err == nil && len(msgs) == 0 {
		t.Fatalf("plain parse %q: got no error; want a missing-operand error", src)
	}
}
