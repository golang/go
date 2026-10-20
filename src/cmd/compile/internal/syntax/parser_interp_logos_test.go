// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"strings"
	"testing"
)

// M6-S1 parser attacks for InterpExpr{Parts []string; Exprs []Expr}
// (SPEC Sec 1.4, T-104.4; logos-docs/IMPLEMENTATION.md).
//
// Offensive-first TDD: each attack below fails before the M6-S1
// implementation (missing diagnostics, missing nodes) and passes after.

func parseInterpLogos(t *testing.T, src string) (*File, []string) {
	t.Helper()
	var msgs []string
	// Unlike parseLogos, this helper tolerates a first error return:
	// error-attack inputs report through msgs, which the caller asserts.
	f, _ := Parse(nil, strings.NewReader(src), func(err error) {
		msgs = append(msgs, err.Error())
	}, nil, CheckBranches|LogosMode)
	return f, msgs
}

func interpOf(t *testing.T, f *File) *InterpExpr {
	t.Helper()
	var found []*InterpExpr
	Inspect(f, func(n Node) bool {
		if e, ok := n.(*InterpExpr); ok {
			found = append(found, e)
		}
		return true
	})
	if len(found) != 1 {
		t.Fatalf("want 1 *InterpExpr, found %d", len(found))
	}
	return found[0]
}

func TestLogosInterpParts(t *testing.T) {
	f, msgs := parseInterpLogos(t, "package p\nfunc f(name string, n int) string {\n\treturn $\"User {name} has {n} items\"\n}\n")
	if len(msgs) != 0 {
		t.Fatalf("unexpected errors: %q", msgs)
	}
	e := interpOf(t, f)
	if len(e.Parts) != 3 || len(e.Exprs) != 2 {
		t.Fatalf("got %d parts %d exprs; want 3 parts 2 exprs", len(e.Parts), len(e.Exprs))
	}
	if e.Parts[0] != "User " || e.Parts[1] != " has " || e.Parts[2] != " items" {
		t.Fatalf("parts = %q", e.Parts)
	}
	if n, ok := e.Exprs[0].(*Name); !ok || n.Value != "name" {
		t.Fatalf("expr0 = %T %v; want *Name name", e.Exprs[0], e.Exprs[0])
	}
	checkPos(t, "InterpExpr.Pos", e.Pos(), 3, 9)
	if len(e.Opens) != 2 || len(e.Closes) != 2 {
		t.Fatalf("got %d opens %d closes; want 2 and 2", len(e.Opens), len(e.Closes))
	}
	// `$"User {name} has {n} items"` starts at 3:9: `{name}` opens at
	// 3:16 and closes at 3:21; `{n}` opens at 3:27 and closes at 3:29.
	checkPos(t, "Opens[0]", e.Opens[0], 3, 16)
	checkPos(t, "Closes[0]", e.Closes[0], 3, 21)
	checkPos(t, "Opens[1]", e.Opens[1], 3, 27)
	checkPos(t, "Closes[1]", e.Closes[1], 3, 29)
}

func TestLogosInterpEscapes(t *testing.T) {
	f, msgs := parseInterpLogos(t, "package p\nfunc f() string {\n\treturn $\"{{literal}}\"\n}\n")
	if len(msgs) != 0 {
		t.Fatalf("unexpected errors: %q", msgs)
	}
	e := interpOf(t, f)
	if len(e.Exprs) != 0 {
		t.Fatalf("got %d exprs; want 0", len(e.Exprs))
	}
	if len(e.Parts) != 1 || e.Parts[0] != "{literal}" {
		t.Fatalf("parts = %q; want [\"{literal}\"]", e.Parts)
	}
}

func TestLogosInterpNestedQuotes(t *testing.T) {
	f, msgs := parseInterpLogos(t, "package p\nfunc f(g func(string) string) string {\n\treturn $\"{g(\"q\")}\"\n}\n")
	if len(msgs) != 0 {
		t.Fatalf("unexpected errors: %q", msgs)
	}
	e := interpOf(t, f)
	if len(e.Exprs) != 1 {
		t.Fatalf("got %d exprs; want 1", len(e.Exprs))
	}
	if _, ok := e.Exprs[0].(*CallExpr); !ok {
		t.Fatalf("expr0 = %T; want *CallExpr", e.Exprs[0])
	}
}

func TestLogosInterpUnterminated(t *testing.T) {
	// The `{` never closes: the scanner completes the token silently at
	// EOF and the parser reports exactly one error at the brace.
	_, msgs := parseInterpLogos(t, "package p\nvar _ = $\"a {x")
	if len(msgs) != 1 {
		t.Fatalf("got %d errors %q; want 1", len(msgs), msgs)
	}
	// With a nil file base the position prints as <unknown position>;
	// the bracketed absolute position and the message carry the check.
	want := "[:2:13]: syntax error: unterminated interpolation"
	if !strings.HasSuffix(msgs[0], want) {
		t.Fatalf("got %q; want suffix %q", msgs[0], want)
	}
}

func TestLogosInterpRoundTrip(t *testing.T) {
	src := "package p\nfunc f(name string) string {\n\treturn $\"User {name} has {{2}} items\"\n}\n"
	f, msgs := parseInterpLogos(t, src)
	if len(msgs) != 0 {
		t.Fatalf("unexpected errors: %q", msgs)
	}
	got := printFull(t, f)
	if !strings.Contains(got, `$"User {name} has {{2}} items"`) {
		t.Fatalf("round trip lost the literal:\n%s", got)
	}
}
