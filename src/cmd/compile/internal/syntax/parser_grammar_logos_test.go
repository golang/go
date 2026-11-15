// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"strings"
	"testing"
)

type grammarProbe struct {
	prod  string
	src   string
	found func(*File) bool
}

func hasNode[N Node](f *File) bool {
	found := false
	Inspect(f, func(n Node) bool {
		if _, ok := n.(N); ok {
			found = true
			return false
		}
		return true
	})
	return found
}

func hasDeferTry(f *File) bool {
	found := false
	Inspect(f, func(n Node) bool {
		if c, ok := n.(*CallStmt); ok && c.Tok == Defer {
			Inspect(c.Call, func(m Node) bool {
				if _, ok := m.(*TryExpr); ok {
					found = true
					return false
				}
				return true
			})
		}
		return true
	})
	return found
}

func hasChainCall(f *File) bool {
	found := false
	Inspect(f, func(n Node) bool {
		if c, ok := n.(*CallExpr); ok {
			if _, ok := Unparen(c.Fun).(*SelectorExpr); ok {
				found = true
				return false
			}
		}
		return true
	})
	return found
}
func hasUnionDecl(f *File) bool {
	found := false
	Inspect(f, func(n Node) bool {
		if it, ok := n.(*InterfaceType); ok && len(it.MethodList) == 1 {
			if fld := it.MethodList[0]; fld.Name == nil {
				if _, ok := Unparen(fld.Type).(*Operation); ok {
					found = true
					return false
				}
			}
		}
		return true
	})
	return found
}

func hasUnionTerms(f *File) bool {
	found := false
	Inspect(f, func(n Node) bool {
		if op, ok := n.(*Operation); ok && op.Op == Or {
			found = true
			return false
		}
		return true
	})
	return found
}
func hasReadonlyMethod(f *File) bool {
	found := false
	Inspect(f, func(n Node) bool {
		if fld, ok := n.(*Field); ok && fld.Const {
			found = true
			return false
		}
		return true
	})
	return found
}

var grammarProbes = []grammarProbe{
	{"PostfixTry", "package p\nfunc f() { g()? }\n", hasNode[*TryExpr]},
	{"OptionType", "package p\nfunc f(x int?) {}\n", hasNode[*OptionType]},
	{"Coalesce", "package p\nfunc f() { _ = a ?? b }\n", hasNode[*CoalesceExpr]},
	{"OptChain", "package p\nfunc f() { _ = a?.b }\n", hasNode[*OptSelectExpr]},
	{"Interp", "package p\nvar s = $\"n={1}\"\n", hasNode[*InterpExpr]},
	{"Lambda", "package p\nfunc f() { g := (a, b) => a }\n", hasNode[*LambdaExpr]},
	{"ChainCall", "package p\nfunc f() {\n_ = xs\n.filter(1)\n.map(2)\n}\n", hasChainCall},
	{"TupleType", "package p\nvar v (int, string)\n", hasNode[*TupleType]},
	{"TupleLit", "package p\nfunc f() { _ = (a, b) }\n", hasNode[*TupleLit]},
	{"SwitchExpr", "package p\nfunc f(v int) { _ = switch v { case 1 => 1; default => 2 } }\n", hasNode[*SwitchExpr]},
	{"Arm", "package p\nfunc f(v int) { _ = switch v { case 1 => 1; default => 2 } }\n", hasNode[*ArmClause]},
	{"UnionDecl", "package p\ntype U interface { int | string }\n", hasUnionDecl},
	{"UnionTerms", "package p\ntype U interface { int | string }\n", hasUnionTerms},
	{"AssertDecl", "package p\ntype I interface { M() }\ntype S struct : I { X int }\n", hasNode[*AssertedStruct]},
	{"PropertyElem", "package p\ntype C interface { const Name string\nHP int }\n", hasNode[*PropertyType]},
	{"ReadonlyMethod", "package p\ntype I interface { const M() string }\n", hasReadonlyMethod},
	{"ConstView", "package p\nfunc f(x const []int) {}\n", hasNode[*ConstType]},
	{"ConstDecl", "package p\ntype S struct{ A int }\nconst K = S{A: 1}\n", hasNode[*ConstDecl]},
	{"ConstIf", "package p\nfunc f() { const if true {} }\n", hasNode[*ConstIfStmt]},
	{"ConstFor", "package p\nfunc f() { const for i := range x {} }\n", hasNode[*ConstForStmt]},
	{"FieldProj", "package p\nfunc f() { _ = v.[f] }\n", hasNode[*FieldProjExpr]},
	{"ConstSwitch", "package p\nfunc f[T any](v T) int {\n_ = v\nconst switch T {\ncase int:\nreturn 1\n}\nreturn 0\n}\n", hasNode[*ConstSwitchStmt]},
	{"DeferTry", "package p\nfunc f() { defer g()? }\n", hasDeferTry},
	{"ErrDefer", "package p\nfunc f() { errdefer g() }\n", hasNode[*ErrdeferStmt]},
	{"ScopeDefer", "package p\nfunc f() { scope defer g() }\n", hasNode[*ScopeDeferStmt]},
	{"GoGroup", "package p\nfunc f(ctx int) { go group (ctx) { go { g() } } }\n", hasNode[*GroupStmt]},
	{"GroupChild", "package p\nfunc f(ctx int) { go group (ctx) { go { g() } } }\n", hasNode[*GroupChild]},
	{"YieldStmt", "package p\nfunc f() { yield 1 }\n", hasNode[*YieldStmt]},
	{"SpreadElem", "package p\ntype S struct{ A, B int }\nvar _ = S{A: 1, ..s}\n", hasNode[*SpreadExpr]},
	{"DefaultField", "package p\ntype S struct { Port int = 8080 }\n", hasNode[*FieldDefault]},
}

func TestGrammarProductionsHaveFixtures(t *testing.T) {
	seen := make(map[string]bool, len(grammarProbes))
	for _, g := range grammarProbes {
		if seen[g.prod] {
			t.Fatalf("duplicate probe for %s", g.prod)
		}
		seen[g.prod] = true
		f := parseLogos(t, g.src)
		if !g.found(f) {
			t.Errorf("%s: no witness node in %q", g.prod, g.src)
		}
	}
}

func TestGrammarConstFuncPositive(t *testing.T) {
	f := parseLogos(t, "package p\nconst func F() {}\n")
	if !hasNode[*ConstFuncDecl](f) {
		t.Fatalf("const func: no *ConstFuncDecl")
	}
	g := parseLogos(t, "package p\nfunc f() { const if true {} }\n")
	if !hasNode[*ConstIfStmt](g) {
		t.Fatalf("const if: no *ConstIfStmt")
	}
	h := parseLogos(t, "package p\nfunc f() { const for i := range x {} }\n")
	if !hasNode[*ConstForStmt](h) {
		t.Fatalf("const for: no *ConstForStmt")
	}
}

func TestGrammarOptionRejectsAndNests(t *testing.T) {
	var msgs []string
	bad, _ := Parse(nil, strings.NewReader("package p\ntype A T??\n"), func(err error) {
		msgs = append(msgs, err.Error())
	}, nil, CheckBranches|LogosMode)
	if len(msgs) == 0 {
		t.Fatalf("T??: parsed without error, want a syntax error")
	}
	found := false
	Inspect(bad, func(n Node) bool {
		if opt, ok := n.(*OptionType); ok {
			if _, ok := Unparen(opt.Elem).(*OptionType); ok {
				found = true
				return false
			}
		}
		return true
	})
	if !found {
		t.Fatalf("T??: no doubly-nested *OptionType recovery in %q", msgs)
	}
	f := parseLogos(t, "package p\nfunc f(x (int?)?) {}\n")
	decl, ok := f.DeclList[0].(*FuncDecl)
	if !ok {
		t.Fatalf("decl 0 is %T, want *FuncDecl", f.DeclList[0])
	}
	if len(decl.Type.ParamList) != 1 {
		t.Fatalf("got %d params, want 1", len(decl.Type.ParamList))
	}
	outer, ok := decl.Type.ParamList[0].Type.(*OptionType)
	if !ok {
		t.Fatalf("param type is %T, want *OptionType", decl.Type.ParamList[0].Type)
	}
	if _, ok := Unparen(outer.Elem).(*OptionType); !ok {
		t.Fatalf("outer elem is %T, want nested *OptionType", outer.Elem)
	}
}
func TestGrammarReadonlyDistinct(t *testing.T) {
	f := parseLogos(t, "package p\ntype I interface { const M() string }\n")
	decl, ok := f.DeclList[0].(*TypeDecl)
	if !ok {
		t.Fatalf("decl 0 is %T, want *TypeDecl", f.DeclList[0])
	}
	it, ok := decl.Type.(*InterfaceType)
	if !ok || len(it.MethodList) != 1 {
		t.Fatalf("interface methods: %+v", decl.Type)
	}
	fld := it.MethodList[0]
	if !fld.Const {
		t.Fatalf("readonly method: Const flag not set")
	}
	if _, ok := fld.Type.(*FuncType); !ok {
		t.Fatalf("readonly method type is %T, want *FuncType", fld.Type)
	}
	g := parseLogos(t, "package p\ntype C interface { const Name string }\n")
	gdecl, ok := g.DeclList[0].(*TypeDecl)
	if !ok {
		t.Fatalf("decl 0 is %T, want *TypeDecl", g.DeclList[0])
	}
	git, ok := gdecl.Type.(*InterfaceType)
	if !ok || len(git.MethodList) != 1 {
		t.Fatalf("property methods: %+v", gdecl.Type)
	}
	prop, ok := git.MethodList[0].Type.(*PropertyType)
	if !ok || !prop.Const {
		t.Fatalf("property elem is %T, want *PropertyType const", git.MethodList[0].Type)
	}
}

func TestGrammarReadonlyMethodOnlyInterface(t *testing.T) {
	tests := []struct {
		name string
		src  string
		pos  string
	}{
		{"receiver method", "package p\ntype T struct{}\nfunc (r T) const M() {}\n", ":3:12"},
		{"method expression", "package p\ntype T struct{}\nfunc (r T) M() {}\nvar _ = T.const M\n", ":4:11"},
		{"function literal", "package p\nfunc f() {\n\tg := const func() {}\n}\n", ":3:20"},
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
	parseLogos(t, "package p\ntype T struct{}\nfunc (r const *T) M() {}\n")
}

func TestGrammarCoalesceShapes(t *testing.T) {
	f := parseLogos(t, "package p\nfunc f(a, b, c int) { _ = a || b ?? c }\n")
	stmts := funcBodyStmts(t, f, "f")
	as, ok := stmts[0].(*AssignStmt)
	if !ok {
		t.Fatalf("first statement is %T, want *AssignStmt", stmts[0])
	}
	top, ok := as.Rhs.(*CoalesceExpr)
	if !ok {
		t.Fatalf("a || b ?? c rhs is %T, want *CoalesceExpr", as.Rhs)
	}
	op, ok := Unparen(top.X).(*Operation)
	if !ok || op.Op != OrOr {
		t.Fatalf("coalesce left is %T, want || operation", top.X)
	}
	g := parseLogos(t, "package p\nfunc f(a, b, c int) { _ = a ?? b ?? c }\n")
	gstmts := funcBodyStmts(t, g, "f")
	gas, ok := gstmts[0].(*AssignStmt)
	if !ok {
		t.Fatalf("first statement is %T, want *AssignStmt", gstmts[0])
	}
	gtop, ok := gas.Rhs.(*CoalesceExpr)
	if !ok {
		t.Fatalf("a ?? b ?? c rhs is %T, want *CoalesceExpr", gas.Rhs)
	}
	if _, ok := Unparen(gtop.Y).(*CoalesceExpr); !ok {
		t.Fatalf("chained ?? right is %T, want right-nested *CoalesceExpr", gtop.Y)
	}
}

type grammarReject struct {
	prod string
	src  string
	at   string
}

var grammarRejects = []grammarReject{
	{"PostfixTry", "package p\nfunc f() { _ = g(?) }\n", ":2:18"},
	{"OptionType", "package p\nfunc f(? int) {}\n", ":2:8"},
	{"Coalesce", "package p\nfunc f() { _ = a ?? }\n", ":2:21"},
	{"OptChain", "package p\nfunc f() { _ = a?. }\n", ":2:20"},
	{"Interp", "package p\nvar s = $'x'\n", ":2:9"},
	{"Lambda", "package p\nfunc f() { g := (a b) => a }\n", ":2:23"},
	{"ChainCall", "package p\nfunc f() {\n_ = xs.filter 1\n}\n", ":3:15"},
	{"TupleType", "package p\nvar v (, int)\n", ":2:8"},
	{"TupleLit", "package p\nfunc f() { _ = (,a) }\n", ":2:17"},
	{"SwitchExpr", "package p\nfunc f(v int) {\n_ = switch v { case 1 => 1; default 2 } }\n", ":3:37"},
	{"Arm", "package p\nfunc f(v int) {\n_ = switch v { case => 1; default => 2 } }\n", ":3:21"},
	{"UnionDecl", "package p\ntype U interface { int | }\n", ":2:26"},
	{"UnionTerms", "package p\ntype U interface { | int }\n", ":2:20"},
	{"AssertDecl", "package p\ntype S struct I { X int }\n", ":2:15"},
	{"PropertyElem", "package p\ntype C interface { const 42 string }\n", ":2:26"},
	{"ReadonlyMethod", "package p\ntype I interface { const () string }\n", ":2:26"},
	{"ConstView", "package p\nfunc f(x const) {}\n", ":2:15"},
	{"ConstDecl", "package p\ntype S struct{ A int }\nconst K S{A: 1}\n", ":3:10"},
	{"ConstIf", "package p\nfunc f() { const if {} }\n", ":2:21"},
	{"ConstFor", "package p\nfunc f() { const for i = range x {} }\n", ":2:24"},
	{"FieldProj", "package p\nfunc f() { _ = v.[] }\n", ":2:19"},
	{"ConstSwitch", "package p\nfunc f[T any](v T) int {\n_ = v\nconst switch {\ncase int:\nreturn 1\n}\nreturn 0\n}\n", ":4:14"},
	{"DeferTry", "package p\nfunc f() { defer ? }\n", ":2:18"},
	{"ErrDefer", "package p\nfunc f() { errdefer 1 }\n", ":2:21"},
	{"ScopeDefer", "package p\nfunc f() { scope defer 42 }\n", ":2:24"},
	{"GoGroup", "package p\nfunc f(ctx int) { go group (ctx) + { go { g() } } }\n", ":2:34"},
	{"GroupChild", "package p\nfunc f(ctx int) { go group (ctx) { go { g( } } }\n", ":2:44"},
	{"YieldStmt", "package p\nfunc f() { yield if x {} }\n", ":2:18"},
	{"SpreadElem", "package p\ntype S struct{ A, B int }\nvar _ = S{A: 1, ..}\n", ":3:19"},
	{"DefaultField", "package p\ntype S struct { Port int = }\n", ":2:28"},
	{"ConstFunc", "package p\nconst func F {}\n", ":2:14"},
}

func TestGrammarProductionsRejectMalformed(t *testing.T) {
	seen := make(map[string]bool, len(grammarRejects))
	for _, g := range grammarRejects {
		if seen[g.prod] {
			t.Fatalf("duplicate malformed probe for %s", g.prod)
		}
		seen[g.prod] = true
		msgs := parseErrs(t, g.src, CheckBranches|LogosMode)
		if len(msgs) == 0 {
			t.Errorf("%s: %q parsed without error, want an error at %s", g.prod, g.src, g.at)
			continue
		}
		if !strings.Contains(msgs[0], g.at) {
			t.Errorf("%s: first error %q does not report at the malformed token %s", g.prod, msgs[0], g.at)
		}
	}
}
