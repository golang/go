// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"strings"
	"testing"
)

func parseErrors(src string, mode Mode) []string {
	var msgs []string
	Parse(nil, strings.NewReader(src), func(err error) {
		msgs = append(msgs, err.Error())
	}, nil, mode)
	return msgs
}

func firstCall(t *testing.T, src string) *CallExpr {
	t.Helper()
	f := parseLogos(t, src)
	stmt, ok := funcBodyStmts(t, f, "f")[0].(*ExprStmt)
	if !ok {
		t.Fatalf("stmt 0 is not an expression statement")
	}
	call, ok := stmt.X.(*CallExpr)
	if !ok {
		t.Fatalf("expression is %T; want *CallExpr", stmt.X)
	}
	return call
}

func namedArg(t *testing.T, e Expr) (string, Expr) {
	t.Helper()
	kv, ok := e.(*KeyValueExpr)
	if !ok {
		t.Fatalf("argument is %T; want *KeyValueExpr", e)
	}
	key, ok := kv.Key.(*Name)
	if !ok {
		t.Fatalf("named argument key is %T; want *Name", kv.Key)
	}
	return key.Value, kv.Value
}

func TestNamedArgParsesInCall(t *testing.T) {
	call := firstCall(t, "package p\nfunc f() { g(1, limit = 8) }\nfunc g(int, int) {}\n")
	if len(call.ArgList) != 2 {
		t.Fatalf("call has %d arguments; want 2", len(call.ArgList))
	}
	if _, ok := call.ArgList[0].(*BasicLit); !ok {
		t.Errorf("argument 0 is %T; want *BasicLit", call.ArgList[0])
	}
	name, value := namedArg(t, call.ArgList[1])
	if name != "limit" {
		t.Errorf("named argument is %q; want limit", name)
	}
	if lit, ok := value.(*BasicLit); !ok || lit.Value != "8" {
		t.Errorf("named argument value is %v; want 8", value)
	}
}

func TestNamedArgPositions(t *testing.T) {
	call := firstCall(t, "package p\nfunc f() { g(1, limit = 8) }\nfunc g(int, int) {}\n")
	arg := call.ArgList[1]
	checkPos(t, "StartPos(named argument)", StartPos(arg), 2, 17)
	checkPos(t, "EndPos(named argument)", EndPos(arg), 2, 26)
}

func TestNamedArgInStageChain(t *testing.T) {
	call := firstCall(t, "package p\nfunc f(xs []int, ctx int) { xs.group(ctx, limit = 8).map(x => x) }\n")
	sel, ok := call.Fun.(*SelectorExpr)
	if !ok {
		t.Fatalf("outer call fun is %T; want *SelectorExpr", call.Fun)
	}
	stage, ok := sel.X.(*CallExpr)
	if !ok {
		t.Fatalf("receiver is %T; want *CallExpr", sel.X)
	}
	if len(stage.ArgList) != 2 {
		t.Fatalf("stage call has %d arguments; want 2", len(stage.ArgList))
	}
	if name, _ := namedArg(t, stage.ArgList[1]); name != "limit" {
		t.Errorf("stage named argument is %q; want limit", name)
	}
}

func TestNamedArgSeveralInOneCall(t *testing.T) {
	call := firstCall(t, "package p\nfunc f() { g(a = 1, b = 2) }\nfunc g(a, b int) {}\n")
	if len(call.ArgList) != 2 {
		t.Fatalf("call has %d arguments; want 2", len(call.ArgList))
	}
	for i, want := range []string{"a", "b"} {
		if name, _ := namedArg(t, call.ArgList[i]); name != want {
			t.Errorf("named argument %d is %q; want %q", i, name, want)
		}
	}
}

func TestNamedArgPlainGoStaysSyntaxError(t *testing.T) {
	src := "package p\nfunc f() { g(1, limit = 8) }\nfunc g(int, int) {}\n"
	if len(parseErrors(src, CheckBranches)) == 0 {
		t.Fatal("plain Go parsed a named argument without error")
	}
}

func TestNamedArgKeyMustBeName(t *testing.T) {
	for _, src := range []string{
		"package p\nfunc f(s struct{ b int }) { g(s.b = 1) }\nfunc g(int) {}\n",
		"package p\nfunc f() { g(1 = 2) }\nfunc g(int) {}\n",
		"package p\nfunc f(xs []int) { g(xs[0] = 2) }\nfunc g(int) {}\n",
	} {
		if len(parseErrors(src, CheckBranches|LogosMode)) == 0 {
			t.Errorf("parsed without error: %q", src)
		}
	}
}

func TestNamedArgLeavesOtherArgumentsAlone(t *testing.T) {
	call := firstCall(t, "package p\nfunc f() { g(a == b, x => x, T{a: 1}, h(c), s...) }\nfunc g(...int) {}\n")
	for i, arg := range call.ArgList {
		if _, ok := arg.(*KeyValueExpr); ok {
			t.Errorf("argument %d became a named argument", i)
		}
	}
	if len(call.ArgList) != 5 {
		t.Errorf("call has %d arguments; want 5", len(call.ArgList))
	}
}
