// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"strings"
	"testing"
)

func parseGroup(t *testing.T, src string) *File {
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

func groupFuncBody(t *testing.T, src string) []Stmt {
	t.Helper()
	f := parseGroup(t, src)
	decl, ok := f.DeclList[0].(*FuncDecl)
	if !ok {
		t.Fatalf("decl 0 is %T; want *FuncDecl", f.DeclList[0])
	}
	return decl.Body.List
}

func TestGroupParsesCtxAndBody(t *testing.T) {
	list := groupFuncBody(t, "package p\nfunc f(ctx int) { go group (ctx) { go { g() } } }\n")
	g, ok := list[0].(*GroupStmt)
	if !ok {
		t.Fatalf("stmt 0 is %T; want *GroupStmt", list[0])
	}
	if _, ok := g.Ctx.(*Name); !ok {
		t.Fatalf("group ctx is %T; want *Name", g.Ctx)
	}
	if g.Limit != nil {
		t.Fatalf("group limit is %v; want nil", g.Limit)
	}
	if len(g.Body.List) != 1 {
		t.Fatalf("group body holds %d statements; want 1", len(g.Body.List))
	}
	c, ok := g.Body.List[0].(*GroupChild)
	if !ok {
		t.Fatalf("body stmt 0 is %T; want *GroupChild", g.Body.List[0])
	}
	if len(c.Body.List) != 1 {
		t.Fatalf("child body holds %d statements; want 1", len(c.Body.List))
	}
	var walked bool
	Inspect(list[0], func(n Node) bool {
		if _, ok := n.(*GroupChild); ok {
			walked = true
		}
		return true
	})
	if !walked {
		t.Fatal("Inspect found no *GroupChild under the group statement")
	}
}

func TestGroupParsesLimit(t *testing.T) {
	list := groupFuncBody(t, "package p\nfunc f(ctx int) { go group (ctx, limit = 8) { } }\n")
	g, ok := list[0].(*GroupStmt)
	if !ok {
		t.Fatalf("stmt 0 is %T; want *GroupStmt", list[0])
	}
	lit, ok := g.Limit.(*BasicLit)
	if !ok {
		t.Fatalf("group limit is %T; want *BasicLit", g.Limit)
	}
	if lit.Value != "8" {
		t.Fatalf("group limit is %q; want 8", lit.Value)
	}
}

func TestGroupCallWithoutBlockStaysCall(t *testing.T) {
	list := groupFuncBody(t, "package p\nfunc f(x int) { go group(x) }\n")
	cs, ok := list[0].(*CallStmt)
	if !ok {
		t.Fatalf("stmt 0 is %T; want *CallStmt", list[0])
	}
	call, ok := Unparen(cs.Call).(*CallExpr)
	if !ok {
		t.Fatalf("go call is %T; want *CallExpr", cs.Call)
	}
	if name, ok := call.Fun.(*Name); !ok || name.Value != "group" {
		t.Fatalf("go call fun is %v; want group", call.Fun)
	}
}

func TestGroupPlainModeKeepsGoCall(t *testing.T) {
	src := "package p\nfunc f(ctx int) { go group (ctx) }\n"
	var msgs []string
	f, err := Parse(nil, strings.NewReader(src), func(err error) {
		msgs = append(msgs, err.Error())
	}, nil, CheckBranches)
	if err != nil {
		t.Fatalf("parse %q: %v", src, err)
	}
	if len(msgs) != 0 {
		t.Fatalf("parse %q: errors %q", src, msgs)
	}
	decl := f.DeclList[0].(*FuncDecl)
	if _, ok := decl.Body.List[0].(*CallStmt); !ok {
		t.Fatalf("plain stmt 0 is %T; want *CallStmt", decl.Body.List[0])
	}
}

func TestGroupBlockNeedsGo(t *testing.T) {
	f := parseGroup(t, "package p\nfunc f(ctx int) { go group (ctx) { g() } }\nfunc g() {}\n")
	decl := f.DeclList[0].(*FuncDecl)
	if _, ok := decl.Body.List[0].(*GroupStmt); !ok {
		t.Fatalf("go group stmt is %T; want *GroupStmt", decl.Body.List[0])
	}
	src := "package p\nfunc f(ctx int) { group (ctx) { g() } }\nfunc g() {}\n"
	var msgs []string
	_, err := Parse(nil, strings.NewReader(src), func(err error) {
		msgs = append(msgs, err.Error())
	}, nil, CheckBranches|LogosMode)
	if err == nil && len(msgs) == 0 {
		t.Fatal("misplaced group block parsed without error")
	}
	if len(msgs) == 0 || !strings.Contains(msgs[0], "unexpected { at end of statement") {
		t.Fatalf("misplaced group first error is %q; want unexpected { at end of statement", msgs)
	}
}
