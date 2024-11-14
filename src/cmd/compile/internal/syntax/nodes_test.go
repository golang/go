// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"fmt"
	"strings"
	"testing"
)

// A test is a source code snippet of a particular node type.
// In the snippet, a '@' indicates the position recorded by
// the parser when creating the respective node.
type test struct {
	nodetyp string
	snippet string
}

var decls = []test{
	// The position of declarations is always the
	// position of the first token of an individual
	// declaration, independent of grouping.
	{"ImportDecl", `import @"math"`},
	{"ImportDecl", `import @mymath "math"`},
	{"ImportDecl", `import @. "math"`},
	{"ImportDecl", `import (@"math")`},
	{"ImportDecl", `import (@mymath "math")`},
	{"ImportDecl", `import (@. "math")`},

	{"ConstDecl", `const @x`},
	{"ConstDecl", `const @x = 0`},
	{"ConstDecl", `const @x, y, z = 0, 1, 2`},
	{"ConstDecl", `const (@x)`},
	{"ConstDecl", `const (@x = 0)`},
	{"ConstDecl", `const (@x, y, z = 0, 1, 2)`},

	{"TypeDecl", `type @T int`},
	{"TypeDecl", `type @T = int`},
	{"TypeDecl", `type (@T int)`},
	{"TypeDecl", `type (@T = int)`},

	{"VarDecl", `var @x int`},
	{"VarDecl", `var @x, y, z int`},
	{"VarDecl", `var @x int = 0`},
	{"VarDecl", `var @x, y, z int = 1, 2, 3`},
	{"VarDecl", `var @x = 0`},
	{"VarDecl", `var @x, y, z = 1, 2, 3`},
	{"VarDecl", `var (@x int)`},
	{"VarDecl", `var (@x, y, z int)`},
	{"VarDecl", `var (@x int = 0)`},
	{"VarDecl", `var (@x, y, z int = 1, 2, 3)`},
	{"VarDecl", `var (@x = 0)`},
	{"VarDecl", `var (@x, y, z = 1, 2, 3)`},

	{"FuncDecl", `func @f() {}`},
	{"FuncDecl", `func @(T) f() {}`},
	{"FuncDecl", `func @(x T) f() {}`},
}

var exprs = []test{
	// The position of an expression is the position
	// of the left-most token that identifies the
	// kind of expression.
	{"Name", `@x`},

	{"BasicLit", `@0`},
	{"BasicLit", `@0x123`},
	{"BasicLit", `@3.1415`},
	{"BasicLit", `@.2718`},
	{"BasicLit", `@1i`},
	{"BasicLit", `@'a'`},
	{"BasicLit", `@"abc"`},
	{"BasicLit", "@`abc`"},

	{"CompositeLit", `@{}`},
	{"CompositeLit", `T@{}`},
	{"CompositeLit", `struct{x, y int}@{}`},

	{"KeyValueExpr", `"foo"@: true`},
	{"KeyValueExpr", `"a"@: b`},

	{"FuncLit", `@func (){}`},
	{"ParenExpr", `@(x)`},
	{"SelectorExpr", `a@.b`},
	{"IndexExpr", `a@[i]`},

	{"SliceExpr", `a@[:]`},
	{"SliceExpr", `a@[i:]`},
	{"SliceExpr", `a@[:j]`},
	{"SliceExpr", `a@[i:j]`},
	{"SliceExpr", `a@[i:j:k]`},

	{"AssertExpr", `x@.(T)`},

	{"Operation", `@*b`},
	{"Operation", `@+b`},
	{"Operation", `@-b`},
	{"Operation", `@!b`},
	{"Operation", `@^b`},
	{"Operation", `@&b`},
	{"Operation", `@<-b`},

	{"Operation", `a @|| b`},
	{"Operation", `a @&& b`},
	{"Operation", `a @== b`},
	{"Operation", `a @+ b`},
	{"Operation", `a @* b`},

	{"CallExpr", `f@()`},
	{"CallExpr", `f@(x, y, z)`},
	{"CallExpr", `obj.f@(1, 2, 3)`},
	{"CallExpr", `func(x int) int { return x + 1 }@(y)`},

	// ListExpr: tested via multi-value const/var declarations
}

var types = []test{
	{"Operation", `@*T`},
	{"Operation", `@*struct{}`},

	{"ArrayType", `@[10]T`},
	{"ArrayType", `@[...]T`},

	{"SliceType", `@[]T`},
	{"DotsType", `@...T`},
	{"StructType", `@struct{}`},
	{"InterfaceType", `@interface{}`},
	{"FuncType", `func@()`},
	{"MapType", `@map[T]T`},

	{"ChanType", `@chan T`},
	{"ChanType", `@chan<- T`},
	{"ChanType", `@<-chan T`},
}

var fields = []test{
	{"Field", `@T`},
	{"Field", `@(T)`},
	{"Field", `@x T`},
	{"Field", `@x *(T)`},
	{"Field", `@x, y, z T`},
	{"Field", `@x, y, z (*T)`},
}

var stmts = []test{
	{"EmptyStmt", `@`},

	{"LabeledStmt", `L@:`},
	{"LabeledStmt", `L@: ;`},
	{"LabeledStmt", `L@: f()`},

	{"BlockStmt", `@{}`},

	// The position of an ExprStmt is the position of the expression.
	{"ExprStmt", `@<-ch`},
	{"ExprStmt", `f@()`},
	{"ExprStmt", `append@(s, 1, 2, 3)`},

	{"SendStmt", `ch @<- x`},

	{"DeclStmt", `@const x = 0`},
	{"DeclStmt", `@const (x = 0)`},
	{"DeclStmt", `@type T int`},
	{"DeclStmt", `@type T = int`},
	{"DeclStmt", `@type (T1 = int; T2 = float32)`},
	{"DeclStmt", `@var x = 0`},
	{"DeclStmt", `@var x, y, z int`},
	{"DeclStmt", `@var (a, b = 1, 2)`},

	{"AssignStmt", `x @= y`},
	{"AssignStmt", `a, b, x @= 1, 2, 3`},
	{"AssignStmt", `x @+= y`},
	{"AssignStmt", `x @:= y`},
	{"AssignStmt", `x, ok @:= f()`},
	{"AssignStmt", `x@++`},
	{"AssignStmt", `a[i]@--`},

	{"BranchStmt", `@break`},
	{"BranchStmt", `@break L`},
	{"BranchStmt", `@continue`},
	{"BranchStmt", `@continue L`},
	{"BranchStmt", `@fallthrough`},
	{"BranchStmt", `@goto L`},

	{"CallStmt", `@defer f()`},
	{"CallStmt", `@go f()`},

	{"ReturnStmt", `@return`},
	{"ReturnStmt", `@return x`},
	{"ReturnStmt", `@return a, b, a + b*f(1, 2, 3)`},

	{"IfStmt", `@if cond {}`},
	{"IfStmt", `@if cond { f() } else {}`},
	{"IfStmt", `@if cond { f() } else { g(); h() }`},
	{"ForStmt", `@for {}`},
	{"ForStmt", `@for { f() }`},
	{"SwitchStmt", `@switch {}`},
	{"SwitchStmt", `@switch { default: }`},
	{"SwitchStmt", `@switch { default: x++ }`},
	{"SelectStmt", `@select {}`},
	{"SelectStmt", `@select { default: }`},
	{"SelectStmt", `@select { default: ch <- false }`},
}

var ranges = []test{
	{"RangeClause", `@range s`},
	{"RangeClause", `i = @range s`},
	{"RangeClause", `i := @range s`},
	{"RangeClause", `_, x = @range s`},
	{"RangeClause", `i, x = @range s`},
	{"RangeClause", `_, x := @range s.f`},
	{"RangeClause", `i, x := @range f(i)`},
}

var guards = []test{
	{"TypeSwitchGuard", `x@.(type)`},
	{"TypeSwitchGuard", `x := x@.(type)`},
}

var cases = []test{
	{"CaseClause", `@case x:`},
	{"CaseClause", `@case x, y, z:`},
	{"CaseClause", `@case x == 1, y == 2:`},
	{"CaseClause", `@default:`},
}

var comms = []test{
	{"CommClause", `@case <-ch:`},
	{"CommClause", `@case x <- ch:`},
	{"CommClause", `@case x = <-ch:`},
	{"CommClause", `@case x := <-ch:`},
	{"CommClause", `@case x, ok = <-ch: f(1, 2, 3)`},
	{"CommClause", `@case x, ok := <-ch: x++`},
	{"CommClause", `@default:`},
	{"CommClause", `@default: ch <- true`},
}

func TestPos(t *testing.T) {
	// TODO(gri) Once we have a general tree walker, we can use that to find
	// the first occurrence of the respective node and we don't need to hand-
	// extract the node for each specific kind of construct.

	testPos(t, decls, "package p; ", "",
		func { f -> f.DeclList[0] },
	)

	// embed expressions in a composite literal so we can test key:value and naked composite literals
	testPos(t, exprs, "package p; var _ = T{ ", " }",
		func { f -> f.DeclList[0].(*VarDecl).Values.(*CompositeLit).ElemList[0] },
	)

	// embed types in a function  signature so we can test ... types
	testPos(t, types, "package p; func f(", ")",
		func { f -> f.DeclList[0].(*FuncDecl).Type.ParamList[0].Type },
	)

	testPos(t, fields, "package p; func f(", ")",
		func { f -> f.DeclList[0].(*FuncDecl).Type.ParamList[0] },
	)

	testPos(t, stmts, "package p; func _() { ", "; }",
		func { f -> f.DeclList[0].(*FuncDecl).Body.List[0] },
	)

	testPos(t, ranges, "package p; func _() { for ", " {} }",
		func { f -> f.DeclList[0].(*FuncDecl).Body.List[0].(*ForStmt).Init.(*RangeClause) },
	)

	testPos(t, guards, "package p; func _() { switch ", " {} }",
		func { f -> f.DeclList[0].(*FuncDecl).Body.List[0].(*SwitchStmt).Tag.(*TypeSwitchGuard) },
	)

	testPos(t, cases, "package p; func _() { switch { ", " } }",
		func { f -> f.DeclList[0].(*FuncDecl).Body.List[0].(*SwitchStmt).Body[0] },
	)

	testPos(t, comms, "package p; func _() { select { ", " } }",
		func { f -> f.DeclList[0].(*FuncDecl).Body.List[0].(*SelectStmt).Body[0] },
	)
}

func testPos(t *testing.T, list []test, prefix, suffix string, extract func(*File) Node) {
	for _, test := range list {
		// complete source, compute @ position, and strip @ from source
		src, index := stripAt(prefix + test.snippet + suffix)
		if index < 0 {
			t.Errorf("missing @: %s (%s)", src, test.nodetyp)
			continue
		}

		// build syntax tree
		file, err := Parse(nil, strings.NewReader(src), nil, nil, 0)
		if err != nil {
			t.Errorf("parse error: %s: %v (%s)", src, err, test.nodetyp)
			continue
		}

		// extract desired node
		node := extract(file)
		if typ := typeOf(node); typ != test.nodetyp {
			t.Errorf("type error: %s: type = %s, want %s", src, typ, test.nodetyp)
			continue
		}

		// verify node position with expected position as indicated by @
		if pos := int(node.Pos().Col()); pos != index+colbase {
			t.Errorf("pos error: %s: pos = %d, want %d (%s)", src, pos, index+colbase, test.nodetyp)
			continue
		}
	}
}

func stripAt(s string) (string, int) {
	if i := strings.Index(s, "@"); i >= 0 {
		return s[:i] + s[i+1:], i
	}
	return s, -1
}

func typeOf(n Node) string {
	const prefix = "*syntax."
	k := fmt.Sprintf("%T", n)
	return strings.TrimPrefix(k, prefix)
}
